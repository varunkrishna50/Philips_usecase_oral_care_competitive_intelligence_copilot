"""Grounded answer generation (3b): ties retrieval (3a) + the
deterministic comparison engine + the LLM (or its deterministic
fallback) + citation validation into one `answer_query()` call.

Design choice worth calling out: comparisons (3B.1) are answered
ENTIRELY by comparison_engine.py's deterministic summaries, never
passed through the LLM at all. The spec is explicit that the LLM must
never do the arithmetic and only "verbalizes" an already-computed
result -- our code already produces that verbalization, so routing it
through the LLM besides would only add hallucination risk for zero
benefit. Normal/exhaustive answers DO use the LLM when available, but
every fact they state must come from evidence we handed it (E1, E2,
... handles mapped back to real evidence_ids before validation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .answer_assembly import cap_citations, drop_redundant_segments, limit_related, prepare_evidence, quote_key
from .attribute_status import (
    attribute_label,
    evidence_ids_for,
    expand_attributes,
    related_evidence,
    resolve_attribute_status,
    status_aware_statement,
)
from .citations import (
    display_section,
    Segment,
    ValidatedSegment,
    assign_markers,
    render_answer_text,
    render_sources_section,
    validate_segments,
)
from .comparison_engine import BOOLEAN_ATTRIBUTES, NUMERIC_ATTRIBUTES, compare_attribute
from .llm_provider import LLMUnavailable, OpenAIStructuredClient
from .models import ProductRecord, RetrievalDocument
from .retrieve import RetrievalEngine, RetrievalResult
from .safety import PROMPT_INJECTION_NOTICE, USER_MESSAGE_TRUST_NOTICE
from .scope import NO_SCOPE_MESSAGE, detect_medical_advice_request, detect_out_of_scope_source, has_no_scoping_signal

DEFAULT_COMPARISON_ATTRIBUTES = sorted(NUMERIC_ATTRIBUTES | BOOLEAN_ATTRIBUTES)


# user-facing text when no segment survives validation
EMPTY_ANSWER_MESSAGE = "The page does not say enough to answer this."


@dataclass
class AnswerResult:
    query: str
    answer_text: str
    sources_text: str
    segments: list[ValidatedSegment] = field(default_factory=list)
    used_llm: bool = False
    llm_unavailable_reason: Optional[str] = None
    retrieval_result: Optional[RetrievalResult] = None


def _build_system_prompt() -> str:
    return (
        "You are a grounded product-comparison assistant for Philips Sonicare and Oral-B electric "
        "toothbrushes. Answer using ONLY the numbered evidence items (E1, E2, ...) given in the user "
        "message -- never your own general knowledge about these or any other products.\n\n"
        f"{PROMPT_INJECTION_NOTICE}\n\n"
        f"{USER_MESSAGE_TRUST_NOTICE}\n\n"
        "Never perform arithmetic or comparisons yourself; any such computation has already been done "
        "elsewhere and will be added to the answer separately. Never state a fact, number, or product name "
        "that is not directly supported by a cited evidence item.\n\n"
        "Answer fully but concisely: lead with the direct answer, then give the specifics the evidence states "
        "(how the feature works, numbers, mode or item names, the page's own wording), one segment per fact, "
        "each citing the handle(s) that support it. If the only evidence is a bare label (for example just the "
        "feature name), say that this is all the page states. Quote numbers, percentages and claim wording "
        "exactly as the evidence has them -- never round, convert or rephrase a figure (write \"5x\" as 5x, "
        "never \"5 times\"; \"700%\" as 700%, never \"seven times\"). When the evidence gives more than one figure "
        "for the same thing, report each of them. Name the product the question is about in your first segment "
        "(even when the evidence sentence itself does not). State each fact once; do not restate a fact an "
        "earlier segment already gave. Use only facts that answer the question, and leave out unrelated details "
        "even if the evidence contains them. Write in plain words.\n\n"
        'Respond with ONLY a single JSON object of '
        'the exact form {"segments": [{"text": "...", "citation_ids": ["E1"]}]} and nothing else -- no '
        "markdown, no [S1]-style markers (the application assigns those itself)."
    )


def _build_evidence_context(citable_evidence: list[RetrievalDocument]) -> tuple[str, dict[str, str]]:
    lines = []
    handle_map: dict[str, str] = {}
    for i, doc in enumerate(citable_evidence, 1):
        handle = f"E{i}"
        handle_map[handle] = doc.evidence_id
        shown_section = display_section(doc)
        section = f" | {shown_section}" if shown_section else ""
        lines.append(f"{handle}: [{doc.product_name} | {doc.attribute}{section}] {doc.shown_text}")
    return "\n".join(lines), handle_map


def _parse_llm_segments(raw: dict, handle_map: dict[str, str]) -> list[Segment]:
    segments: list[Segment] = []
    items = raw.get("segments") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return segments
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        handles = item.get("citation_ids")
        if not isinstance(text, str) or not text.strip() or not isinstance(handles, list):
            continue
        evidence_ids = [handle_map[h] for h in handles if isinstance(h, str) and h in handle_map]
        if not evidence_ids:
            continue  # every handle hallucinated/unknown -- drop rather than guess
        segments.append(Segment(text=text, citation_ids=evidence_ids, trusted=False))
    return segments


def _fallback_segments_from_evidence(citable_evidence: list[RetrievalDocument]) -> list[Segment]:
    """One segment per distinct (product, evidence_text) pair. Step 2
    occasionally produces two separate, individually-correct evidence
    records with identical text for the same product+attribute (e.g. the
    same package-contents line found under both a TextFeatureN and a
    Shop-By-Feature label) -- without this, the deterministic fallback
    would visibly repeat the same sentence twice. Both evidence_ids are
    still cited.
    """
    # A bare label ("Pressure Sensor", 3 words or fewer) adds nothing once the same product
    # has a fuller sentence for the topic; keep it only when it is all there is.
    richer_products = {d.product_id for d in citable_evidence if len(d.shown_text.split()) > 3}
    citable_evidence = [
        d for d in citable_evidence
        if not (len(d.shown_text.split()) <= 3 and d.product_id in richer_products)
    ]
    order: list[tuple[str, str]] = []
    grouped: dict[tuple[str, str], list[RetrievalDocument]] = {}
    for doc in citable_evidence:
        key = (doc.product_id, doc.shown_text)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(doc)

    segments = []
    for key in order:
        docs = grouped[key]
        segments.append(
            Segment(
                text=f"{docs[0].product_name}: {docs[0].shown_text}",
                citation_ids=[d.evidence_id for d in docs],
                trusted=True,
            )
        )
    return segments


def _call_llm_for_segments(
    instruction: str,
    query: str,
    citable_evidence: list[RetrievalDocument],
    llm_client: Optional[OpenAIStructuredClient],
) -> tuple[list[Segment], Optional[str]]:
    """Returns (segments, failure_reason). failure_reason is None on
    success; segments is empty on failure (caller decides the fallback)."""
    context_text, handle_map = _build_evidence_context(citable_evidence)
    client = llm_client or OpenAIStructuredClient()
    user_prompt = (
        f"{instruction}\n\nQuestion: {query}\n\nEvidence:\n{context_text}\n\n"
        'Return JSON: {"segments": [{"text": "...", "citation_ids": ["E1"]}, ...]}. '
        "Use ONLY the evidence handles listed above. Do not state anything beyond what they say."
    )
    try:
        raw = client.generate_json(_build_system_prompt(), user_prompt)
    except LLMUnavailable as exc:
        return [], f"LLM generation unavailable; returned deterministic grounded response. ({exc})"
    segments = _parse_llm_segments(raw, handle_map)
    if not segments:
        return [], "LLM returned no usable segments; returned deterministic grounded response."
    return segments, None


def answer_query(
    query: str,
    engine: RetrievalEngine,
    llm_client: Optional[OpenAIStructuredClient] = None,
    top_k: int = 10,
    alpha: float = 0.5,
    no_llm: bool = False,
) -> AnswerResult:
    out_of_scope_message = detect_out_of_scope_source(query)
    if out_of_scope_message:
        return AnswerResult(query, out_of_scope_message, "")

    medical_advice_message = detect_medical_advice_request(query)
    if medical_advice_message:
        return AnswerResult(query, medical_advice_message, "")

    retrieval_result = engine.retrieve(query, top_k=top_k, alpha=alpha)
    evidence_by_id = {d.evidence_id: d for d in engine.retrieval_documents}

    if retrieval_result.clarification_needed:
        return AnswerResult(query, retrieval_result.clarification_message, "", retrieval_result=retrieval_result)
    if retrieval_result.not_in_corpus:
        return AnswerResult(query, retrieval_result.not_in_corpus_message, "", retrieval_result=retrieval_result)

    if retrieval_result.exhaustive_matches is not None:
        return _answer_exhaustive(query, retrieval_result, engine, evidence_by_id, llm_client, no_llm)
    if retrieval_result.comparison_results is not None:
        return _answer_comparison(query, retrieval_result, engine, evidence_by_id)
    return _answer_normal(query, retrieval_result, engine, evidence_by_id, llm_client, no_llm)


def _finish(
    query: str, validated: list[ValidatedSegment], evidence_by_id: dict[str, RetrievalDocument],
    used_llm: bool, reason: Optional[str], retrieval_result: RetrievalResult,
) -> AnswerResult:
    validated = drop_redundant_segments(validated, evidence_by_id)
    marker_by_id = assign_markers(validated, evidence_by_id)
    answer_text = render_answer_text(validated, marker_by_id)
    sources_text = render_sources_section(marker_by_id, evidence_by_id)
    if not answer_text:
        answer_text = EMPTY_ANSWER_MESSAGE
    return AnswerResult(
        query=query, answer_text=answer_text, sources_text=sources_text, segments=validated,
        used_llm=used_llm, llm_unavailable_reason=reason, retrieval_result=retrieval_result,
    )


def _answer_normal(
    query: str, retrieval_result: RetrievalResult, engine: RetrievalEngine,
    evidence_by_id: dict[str, RetrievalDocument], llm_client, no_llm: bool,
) -> AnswerResult:
    trusted_segments: list[Segment] = []
    citable_evidence: list[RetrievalDocument] = []
    allowed_product_ids: Optional[set[str]] = None
    # citing a detail attribute of an asked attribute is valid; unrelated attributes still are not
    allowed_attributes: Optional[list[str]] = expand_attributes(retrieval_result.attributes) or None

    asked = list(dict.fromkeys(retrieval_result.attributes))
    scores = {sd.document.evidence_id: sd.hybrid_score for sd in retrieval_result.results}
    if retrieval_result.product_match.kind == "matched" and retrieval_result.attributes:
        product = retrieval_result.product_match.candidates[0]
        allowed_product_ids = {product.product_id}
        for attribute in asked:
            attr_status = resolve_attribute_status(product, attribute, engine.retrieval_documents)
            if attr_status.status == "stated" and attr_status.value is not None:
                # the detail records that answer the question (how it works, numbers, names) first, then the bare label
                related = limit_related(related_evidence(product, attribute, engine.retrieval_documents), query, scores)
                for doc in related + attr_status.evidence:
                    if doc not in citable_evidence:
                        citable_evidence.append(doc)
            else:
                # not_mentioned / stated_absent / ambiguous / stated-null:
                # a deterministic, trusted disclosure -- never asked of the LLM.
                trusted_segments.append(
                    Segment(text=status_aware_statement(attr_status), citation_ids=evidence_ids_for(attr_status), trusted=True)
                )
    elif has_no_scoping_signal(retrieval_result.product_match.kind, retrieval_result.attributes):
        # Neither a specific product nor a specific documented attribute was
        # identified -- retrieval wasn't scoped by either, so its top-k
        # results aren't reliably relevant. Dumping them as a pseudo-answer
        # risks surfacing irrelevant or raw/unpolished evidence text; an
        # honest scope statement is safer. (Also covers open-ended
        # advice-style questions, which this corpus cannot ground anyway.)
        return AnswerResult(query, NO_SCOPE_MESSAGE, "", retrieval_result=retrieval_result)
    else:
        citable_evidence = [sd.document for sd in retrieval_result.results]

    # one record per distinct quote; a quote a code-written sentence already shows is not offered again
    already_quoted = {quote_key(evidence_by_id[c]) for seg in trusted_segments for c in seg.citation_ids if c in evidence_by_id}
    citable_evidence, _aliases = prepare_evidence(citable_evidence, asked, already_quoted)

    llm_segments: list[Segment] = []
    used_llm = False
    reason: Optional[str] = None
    if citable_evidence:
        if no_llm:
            llm_segments = _fallback_segments_from_evidence(citable_evidence)
            reason = "Deterministic mode requested (--no-llm); returned deterministic grounded response."
        else:
            llm_segments, failure = _call_llm_for_segments(
                "Answer the user's question using the evidence below. Create one segment per distinct fact, "
                "each citing the evidence handle(s) that support it.",
                query, citable_evidence, llm_client,
            )
            if failure:
                llm_segments = _fallback_segments_from_evidence(citable_evidence)
                reason = failure
            else:
                used_llm = True

    # at most two citations per segment (direct records first, never fewer than the figures need);
    # the citation validator then runs on the trimmed segments
    llm_segments = [cap_citations(seg, evidence_by_id, asked) for seg in llm_segments]
    validated = validate_segments(trusted_segments + llm_segments, evidence_by_id, engine.product_corpus, allowed_product_ids, allowed_attributes)
    return _finish(query, validated, evidence_by_id, used_llm, reason, retrieval_result)


def _answer_comparison(
    query: str, retrieval_result: RetrievalResult, engine: RetrievalEngine,
    evidence_by_id: dict[str, RetrievalDocument],
) -> AnswerResult:
    product_ids = list(retrieval_result.comparison_results.keys())
    products = [p for p in engine.product_corpus if p.product_id in product_ids]
    if len(products) < 2:
        return _answer_normal(query, retrieval_result, engine, evidence_by_id, None, no_llm=True)

    attributes = list(dict.fromkeys(retrieval_result.attributes)) or DEFAULT_COMPARISON_ATTRIBUTES
    comparisons = [compare_attribute(products[0], products[1], attribute, engine.retrieval_documents) for attribute in attributes]
    segments = [
        Segment(text=comp.summary, citation_ids=comp.citation_evidence_ids, trusted=True)
        for comp in comparisons
    ]
    validated = validate_segments(segments, evidence_by_id, engine.product_corpus)
    reason = "Comparisons are answered deterministically; no arithmetic is delegated to the LLM (3B.1)."
    return _finish(query, validated, evidence_by_id, used_llm=False, reason=reason, retrieval_result=retrieval_result)


def _answer_exhaustive(
    query: str, retrieval_result: RetrievalResult, engine: RetrievalEngine,
    evidence_by_id: dict[str, RetrievalDocument], llm_client, no_llm: bool,
) -> AnswerResult:
    matches = retrieval_result.exhaustive_matches
    if not matches:
        return _finish(query, [], evidence_by_id, used_llm=False, reason="No products match.", retrieval_result=retrieval_result)

    expected_product_ids = {m.product.product_id for m in matches}
    fallback_segments = [
        Segment(
            text=f"{m.product.product_name} has {attribute_label(m.evidence[0].attribute) if m.evidence else 'this attribute'}.",
            citation_ids=[d.evidence_id for d in m.evidence],
            trusted=True,
        )
        for m in matches
    ]

    if no_llm:
        validated = validate_segments(fallback_segments, evidence_by_id, engine.product_corpus)
        reason = "Deterministic mode requested (--no-llm); returned deterministic grounded response."
        return _finish(query, validated, evidence_by_id, used_llm=False, reason=reason, retrieval_result=retrieval_result)

    citable_evidence = [d for m in matches for d in m.evidence]
    llm_segments, failure = _call_llm_for_segments(
        f"List EVERY one of the {len(matches)} products below that match the question -- all must be mentioned, "
        "each in its own segment with its own citation(s). Do not summarize or omit any.",
        query, citable_evidence, llm_client,
    )
    if not failure:
        validated_llm = validate_segments(llm_segments, evidence_by_id, engine.product_corpus)
        cited_product_ids = {
            evidence_by_id[cid].product_id for seg in validated_llm if seg.valid for cid in seg.citation_ids
        }
        if cited_product_ids == expected_product_ids:
            return _finish(query, validated_llm, evidence_by_id, used_llm=True, reason=None, retrieval_result=retrieval_result)
        failure = (
            f"LLM output did not cite all {len(expected_product_ids)} matching products "
            f"(got {len(cited_product_ids)}); returned deterministic grounded response for completeness (3A.13/3B.9)."
        )

    validated = validate_segments(fallback_segments, evidence_by_id, engine.product_corpus)
    return _finish(query, validated, evidence_by_id, used_llm=False, reason=failure, retrieval_result=retrieval_result)
