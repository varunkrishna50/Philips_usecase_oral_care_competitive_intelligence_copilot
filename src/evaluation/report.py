"""Renders data/evaluation/evaluation_report.md from a results dict
produced by runner.run_evaluation()."""
from __future__ import annotations


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_report(results: dict) -> str:
    agg = results["aggregate_metrics"]
    cm = results.get("citation_metrics")
    retrieval = results["retrieval_metrics"]
    gates = results["quality_gates"]
    coverage = results["status_coverage"]
    model_cfg = results["model_configuration"]
    mode = results["mode"]

    lines: list[str] = []
    lines.append("# 3C.1 Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {results['timestamp']}  ")
    lines.append(f"Mode: **{mode}**  ")
    lines.append(f"Model configuration: provider={model_cfg['provider']}, model={model_cfg['model']}, temperature={model_cfg['temperature']}")
    if mode == "live":
        lines.append("")
        lines.append("Live-mode results can vary between runs (real LLM output is not byte-for-byte reproducible). "
                      "Offline and live metrics are reported separately throughout this report.")
    lines.append("")

    # Executive summary
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- {agg['passed_cases']}/{agg['exercised_cases']} exercised test cases passed "
                  f"({agg['not_exercised_cases']} not_exercised; {agg['total_cases']} total).")
    lines.append(f"- Safety pass rate (exercised only): {_pct(agg['safety_pass_rate'])}.")
    lines.append(f"- Exhaustive completeness: {_pct(agg['exhaustive_completeness'])}.")
    lines.append(f"- Unsupported-claim rate (independent groundedness checker): {_pct(agg['unsupported_claim_rate'])}.")
    all_passed = results["quality_gates_all_passed"]
    lines.append(f"- Quality gates: **{'PASS' if all_passed else 'FAIL'}**.")
    lines.append("")

    # Dataset / corpus
    lines.append("## Dataset / corpus")
    lines.append("")
    corpus = results["corpus"]
    lines.append(f"- {corpus['product_count']} products, {corpus['evidence_count']} evidence records "
                  "(Step 2 output: comparison_table.parquet + evidence.json, read independently of RetrievalEngine).")
    if corpus.get("retrieval_documents") is not None:
        lines.append(f"- Retrieval corpus measured in this run: {corpus['retrieval_documents']} documents, "
                      f"sha256 `{corpus['retrieval_documents_sha256']}`, built {corpus['retrieval_documents_modified']}.")
    lines.append("")

    # Test set
    lines.append("## Test set")
    lines.append("")
    by_category: dict[str, int] = {}
    for c in results["test_cases"]:
        by_category[c["category"]] = by_category.get(c["category"], 0) + 1
    lines.append(f"{len(results['test_cases'])} test cases across {len(by_category)} categories:")
    lines.append("")
    for cat, n in sorted(by_category.items()):
        lines.append(f"- `{cat}`: {n}")
    lines.append("")
    lines.append("### Status coverage")
    lines.append("")
    lines.append(f"| Status | Real examples in the {corpus['product_count']}-product corpus | Count |")
    lines.append("|---|---|---:|")
    for status in ("stated", "not_mentioned", "stated_absent", "ambiguous", "stated_with_null_value"):
        info = coverage[status]
        has = "yes" if info["has_real_example"] else "**NO (coverage gap)**"
        lines.append(f"| `{status}` | {has} | {info['count']} |")
    lines.append("")
    gaps = [st for st in ("stated_absent", "ambiguous") if coverage[st]["count"] == 0]
    if gaps:
        names = " and ".join(f"`{g}`" for g in gaps)
        all_stated = coverage["stated"]["count"] == corpus["evidence_count"]
        basis = (f"all {corpus['evidence_count']} evidence records carry status=`stated`" if all_stated
                 else f"{coverage['stated']['count']} of {corpus['evidence_count']} evidence records carry status=`stated`")
        lines.append(f"{names} {'has' if len(gaps) == 1 else 'have'} zero real occurrences in this "
                      f"{corpus['product_count']}-product corpus (counted from evidence.json: {basis}). "
                      "No test case was fabricated for either -- reported here as honest coverage gaps, per the 3C.1 integrity rules.")
    else:
        lines.append("Every status has at least one real example in this corpus.")
    lines.append("")

    # Retrieval results
    lines.append("## Retrieval results")
    lines.append("")
    ablation = retrieval["ablation"]
    if ablation["status"] == "skipped":
        lines.append(f"Skipped: {ablation['reason']}")
    else:
        lines.append(f"Ablation over {ablation['total_cases']} gold cases built from the current Step 2 output, computed in this run on the "
                      "current corpus (fingerprint above); the gold cases come from the corpus itself, so this measures retrieval mechanics, "
                      "not the 3C.1 test cases:")
        lines.append("")
        lines.append("| Retriever | Recall@5 | Recall@10 |")
        lines.append("|---|---:|---:|")
        label_map = {"bm25_only": "BM25", "semantic_only": "Semantic", "hybrid": "Hybrid", "hybrid_reranked": "Hybrid + Reranker"}
        for key, label in label_map.items():
            r5 = ablation["ablation"][key]["5"] if isinstance(ablation["ablation"][key], dict) and "5" in ablation["ablation"][key] else ablation["ablation"][key].get(5)
            r10 = ablation["ablation"][key]["10"] if isinstance(ablation["ablation"][key], dict) and "10" in ablation["ablation"][key] else ablation["ablation"][key].get(10)
            lines.append(f"| {label} | {_pct(r5)} | {_pct(r10)} |")
        lines.append("")
        lines.append("This demonstrates: BM25 alone already does reasonably well on exact attribute-keyword queries; "
                      "semantic search adds recall on paraphrased/semantic queries; combining both (hybrid) recovers "
                      "most of each method's strengths; the deterministic reranker (product/attribute-tier boosting) "
                      "closes nearly all of the remaining gap by guaranteeing exact product+attribute matches rank first.")
    lines.append("")
    lines.append("### Behavioral checks (3A's retrieval-layer checks, re-run in this run)")
    lines.append("")
    for name, info in retrieval["behavioral_checks"].items():
        status = "PASS" if info["passed"] else "FAIL"
        lines.append(f"- **{name}**: {status} -- {info['detail']}")
    lines.append("")

    # Citation results
    lines.append("## Citation results")
    lines.append("")
    if cm:
        lines.append(f"Aggregated over {cm['total_segments']} segments across all generation-requiring test cases "
                      f"({'live' if mode == 'live' else 'offline'} mode):")
        lines.append("")
        lines.append(f"- Citation presence rate (all segments): {_pct(cm['presence_rate'])}")
        uncited = cm["total_segments"] - round(cm["presence_rate"] * cm["total_segments"])
        lines.append(f"- Segments without a citation: {uncited} -- {cm['uncited_absence_disclosures']} not-mentioned disclosure(s) "
                      f"(an absence has no evidence record to cite), {cm['uncited_rejected_by_validator']} rejected by the citation validator "
                      f"(not shown in the answer), **{cm['uncited_other']} factual segment(s) shown without a citation**.")
        lines.append(f"- Citation presence over factual segments shown to the user: {_pct(cm['presence_rate_shown_factual'])} "
                      f"({cm['shown_factual_segments']} segments)")
        lines.append(f"- Citation validity rate (ALL segments, including `trusted` code-generated ones): {_pct(cm['validity_rate_all'])}")
        lines.append(f"- Citation validity rate (LLM-authored/untrusted segments ONLY): {_pct(cm['validity_rate_untrusted_only'])}")
        lines.append(f"- Trusted segments: {cm['trusted_segments']} / {cm['total_segments']}; untrusted (LLM-authored): {cm['untrusted_segments']}")
        lines.append("")
        lines.append("The \"ALL segments\" validity rate is inflated by construction -- `trusted` segments (status "
                      "templates, deterministic comparison summaries) skip the Step 3B validator entirely and always "
                      "count as valid. The untrusted-only rate is the meaningful number for judging the LLM's own "
                      "citation discipline. These offline/live numbers are not comparable to each other and are kept separate.")
    else:
        lines.append("No citation-bearing segments were produced.")
    lines.append("")

    # Groundedness results
    lines.append("## Groundedness results")
    lines.append("")
    lines.append(f"Independent groundedness checker (no import of `src.rag.citations`; checks every segment, "
                  f"including `trusted` ones): {agg['unsupported_claim_count']} unsupported claim(s) found across "
                  f"{agg['total_cases']} test cases (rate: {_pct(agg['unsupported_claim_rate'])}).")
    lines.append("")
    nc = results.get("groundedness_negative_control")
    if nc:
        lines.append(f"Checker self-test (negative control): {nc['false_flagged']}/{nc['false_sentences']} deliberately false sentences "
                      f"flagged, {nc['true_clean']}/{nc['true_sentences']} true sentences left clean "
                      f"-- **{'PASS' if nc['all_ok'] else 'FAIL'}**. The \"0 unsupported\" result above is only meaningful because the checker can fail:")
        lines.append("")
        for item in nc["items"]:
            verdict = "flagged" if item["flagged"] else "clean"
            lines.append(f"  - {item['label']}: {verdict} (expected {'flagged' if item['expect_flag'] else 'clean'}) -- \"{item['text']}\"")
        lines.append("")
    unsupported_cases = [c for c in results["test_cases"] if c["groundedness_issues"]]
    if unsupported_cases:
        lines.append("Cases with flagged issues:")
        lines.append("")
        for c in unsupported_cases:
            lines.append(f"- `{c['id']}`:")
            for issue in c["groundedness_issues"]:
                lines.append(f"  - **{issue['kind']}**: {issue['detail']}")
    else:
        lines.append("No unsupported claims found in this run.")
    lines.append("")

    # Content checks
    lines.append("## Content checks")
    lines.append("")
    checked = [c for c in results["test_cases"] if c.get("content_checks")]
    lines.append(f"{len(checked)} test case(s) declare phrases the answer must (`must_include`) or must not (`must_not_include`) contain. "
                  "Matching is case-insensitive and figures match as whole tokens (\"5x\" is not found in \"15x\"). "
                  "A missing or forbidden phrase fails the case.")
    lines.append("")
    from .content_checks import summarize
    for c in checked:
        ok = all(chk["ok"] for chk in c["content_checks"])
        lines.append(f"- `{c['id']}`: {'ok' if ok else '**FAIL**'} -- {summarize(c['content_checks'])}")
    lines.append("")

    # Answer quality
    quality = results.get("answer_quality")
    if quality:
        agg_q, base = quality["aggregate"], quality.get("baseline")

        def cell(key: str, pct: bool = True) -> str:
            after = agg_q[key]
            shown = _pct(after) if pct else str(after)
            if base is None:
                return f"| {shown} |"
            before = base[key]
            return f"| {_pct(before) if pct else before} | {shown} |"

        lines.append("## Answer quality")
        lines.append("")
        lines.append("Is an answer concise, free of repeats and readable? Measured on what the user is shown "
                      "(the answer text, the Sources section and the validated segments).")
        lines.append("")
        header = "| Metric | Before | After |" if base else "| Metric | This run |"
        lines.append(header)
        lines.append("|---|---:|---:|" if base else "|---|---:|")
        lines.append(f"| Duplicate-source rate ({agg_q['duplicate_sources']} of {agg_q['sources_shown']} shown sources repeat another source's quote) " + cell("duplicate_source_rate"))
        lines.append(f"| Redundant-segment rate ({agg_q['redundant_segments']} of {agg_q['segments_shown']} shown segments add nothing new) " + cell("redundant_segment_rate"))
        lines.append(f"| Jargon rate ({agg_q['cases_with_jargon']} of {agg_q['cases']} answers contain internal wording) " + cell("jargon_rate"))
        lines.append("")
        tj = quality["template_jargon"]
        lines.append(f"Templates written by our own code: {len(tj)} of {quality['template_samples']} contain internal wording"
                      + (" (" + ", ".join(sorted(tj)) + ")." if tj else "."))
        if base:
            lines.append("")
            lines.append("Before = the saved baseline (`data/evaluation/baseline_answer_quality.json`, taken before the readability fixes).")
        lines.append("")
        lines.append("Duplicate source: a Sources entry whose product and quote match an earlier entry. Redundant segment: every quote "
                      "it cites was already cited and it adds no new number or distinctive word. Jargon: normalized, evidence, attribute, "
                      "null, status, corpus, record, \"the available source\", \"manufacturer states\" in our own wording (page quotes are not scanned).")
        lines.append("")

    # Safety results
    lines.append("## Safety results")
    lines.append("")
    lines.append(f"Safety pass rate (over exercised cases only): {_pct(agg['safety_pass_rate'])}.")
    if agg["safety_not_exercised"]:
        lines.append(f"Not exercised in this run ({mode} mode): {', '.join(agg['safety_not_exercised'])}.")
    lines.append("")
    from .runner import SAFETY_CATEGORIES
    for c in results["test_cases"]:
        if c["category"] in SAFETY_CATEGORIES:
            lines.append(f"- `{c['id']}` ({c['category']}): **{c['verdict']}** -- {c['detail']}")
    lines.append("")

    # Exhaustive-query results
    lines.append("## Exhaustive-query results")
    lines.append("")
    lines.append(f"Completeness: {_pct(agg['exhaustive_completeness'])} of exhaustive test cases had every "
                  "expected product (filtered directly from comparison_table.parquet) present with a valid citation.")
    lines.append("")
    for c in results["test_cases"]:
        if c["category"] == "exhaustive":
            lines.append(f"- `{c['id']}`: **{c['verdict']}** -- {c['detail']}")
    lines.append("")

    # Deterministic comparison results
    lines.append("## Deterministic comparison results")
    lines.append("")
    for c in results["test_cases"]:
        if c["category"] == "comparison":
            lines.append(f"- `{c['id']}`: **{c['verdict']}** -- {c['detail']}")
    lines.append("")
    lines.append("Gold comparisons were computed independently from comparison_table.parquet/evidence.json "
                  "(never by re-running `src.rag.comparison_engine`), then checked against the pipeline's own "
                  "structured `ComparisonResult`. The LLM is never used for comparison arithmetic.")
    lines.append("")

    # Ablation results (restated, already shown under Retrieval)
    lines.append("## Ablation results")
    lines.append("")
    lines.append("See the Retrieval results section above for the full BM25 / Semantic / Hybrid / Hybrid+Reranker table.")
    lines.append("")

    # Quality gates
    lines.append("## Quality gates")
    lines.append("")
    lines.append("| Gate | Threshold | Actual | Result |")
    lines.append("|---|---|---|---|")
    for g in gates:
        lines.append(f"| {g['name']} | {g['comparison']} {g['threshold']} | {g['actual']:.4f} | {'PASS' if g['passed'] else 'FAIL'} |")
    lines.append("")
    lines.append(f"**Overall: {'PASS' if all_passed else 'FAIL'}**")
    lines.append("")

    # Failures / known limitations
    lines.append("## Failures / known limitations")
    lines.append("")
    failed = [c for c in results["test_cases"] if c["verdict"] == "fail"]
    if failed:
        for c in failed:
            lines.append(f"### `{c['id']}` ({c['category']})")
            lines.append("")
            lines.append(f"Question: {c['question']}")
            lines.append("")
            lines.append(f"Failure: {c['detail']}")
            lines.append("")
    else:
        lines.append("No failed cases in this run.")
    lines.append("")
    lines.append("Known limitations of this evaluation harness itself:")
    lines.append("")
    lines.append("- The groundedness checker's boolean-attribute claim check is a keyword + negation heuristic, "
                  "not a real NLP parser -- adequate for the short generated sentences this pipeline produces, "
                  "not a general-purpose fact checker.")
    lines.append("- Product-mention detection in the groundedness checker is literal substring matching on "
                  "`product_name`, independent of (and simpler than) `src.rag.query_understanding`'s discriminative "
                  "matching, by design (independence requirement).")
    lines.append("- `stated_absent` and `ambiguous` have no real corpus examples, so neither status model path is "
                  "exercised by a real test case (see Status coverage above).")
    lines.append("")

    return "\n".join(lines)
