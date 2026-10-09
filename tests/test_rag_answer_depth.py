"""Answer depth: an attribute question must reach the detail records that explain it
(pressure_sensing -> pressure_sensing_details, ...), the quotes shown under Sources must be
readable, and figures in an LLM answer must be quoted exactly.

Uses the REAL corpus (tests/conftest.py `real_engine`) and a fake LLM client (no API key needed)."""
from __future__ import annotations

import json
import re

import pytest

from src.rag.attribute_status import ATTRIBUTE_LABELS, RELATED_ATTRIBUTES, expand_attributes
from src.rag.citations import Segment, render_sources_section, validate_segment
from src.rag.config import COMPARISON_TABLE_PARQUET
from src.rag.generation import answer_query
from src.rag.models import RetrievalDocument
from src.rag.query_understanding import detect_attributes
from src.rag.readable import readable_evidence

Q_4100 = "What pressure-sensing feature does the Philips Sonicare 4100 have?"


class FakeLLMClient:
    """Cites every evidence handle it is given, one segment per handle, using the evidence text."""

    def __init__(self):
        self.calls = []

    def generate_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        segments = []
        for m in re.finditer(r"^(E\d+): \[[^\]]*\] (.*)$", user_prompt, re.M):
            segments.append({"text": m.group(2), "citation_ids": [m.group(1)]})
        return {"segments": segments}


def _doc(attribute, text, product_id="p1", name="Sonicare Test 4100", display=""):
    return RetrievalDocument(
        document_id=f"doc:{attribute}", evidence_id=f"{product_id}:{attribute}:0", product_id=product_id,
        brand="Philips", product_name=name, attribute=attribute, value=text, status="stated",
        evidence_text=text, source_url="https://www.usa.philips.com/c-p/X", source_section="Some section",
        source_file="f.html", display_text=display,
    )


# --- the screenshot question ---------------------------------------------------------------------------

def test_pressure_question_without_llm_now_gives_the_details(real_engine):
    result = answer_query(Q_4100, real_engine, no_llm=True)
    text = result.answer_text
    assert "haptic feedback" in text and "gentle vibrations" in text
    cited = {cid for seg in result.segments if seg.valid for cid in seg.citation_ids}
    assert any(":pressure_sensing_details:" in c for c in cited)
    assert "Pressure Sensor [S1]" not in text  # not just the bare label
    assert result.retrieval_result.attributes == ["pressure_sensing"]  # asked list is unchanged


def test_pressure_question_with_llm_sees_and_may_cite_the_detail_records(real_engine):
    client = FakeLLMClient()
    result = answer_query(Q_4100, real_engine, llm_client=client)
    assert result.used_llm
    prompt = client.calls[0][1]
    assert "haptic feedback" in prompt and "gentle vibrations" in prompt
    assert all(seg.valid for seg in result.segments), [s.reasons for s in result.segments if not s.valid]
    assert "Help protect your gums with our pressure sensor" in prompt  # section shown to the model


def test_citing_an_unrelated_attribute_is_still_rejected(real_engine):
    client = FakeLLMClient()

    def bad_response(system_prompt, user_prompt):
        return {"segments": [{"text": "It costs $49.96.", "citation_ids": ["E1"]}]}

    # a price record of the same product is not one of the requested attributes' family
    product = next(p for p in real_engine.product_corpus if p.product_id == "philips_hx4031-71")
    price_doc = next(d for d in real_engine.retrieval_documents if d.product_id == product.product_id and d.attribute == "price")
    seg = Segment(text="It costs $49.96.", citation_ids=[price_doc.evidence_id])
    by_id = {d.evidence_id: d for d in real_engine.retrieval_documents}
    result = validate_segment(seg, by_id, real_engine.product_corpus, {product.product_id},
                              expand_attributes(["pressure_sensing"]))
    assert not result.valid and any("not in requested" in r for r in result.reasons)
    del client, bad_response


# --- every family reaches its details ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, detail_attribute",
    [
        ("Does the Philips Sonicare DiamondClean Smart 9300 have an app?", "app_connectivity_details"),
        ("What does the Philips Sonicare Prestige 9900 have for battery life?", "battery_details"),
        ("How does the timer work on the Philips Sonicare 4100?", "timer_description"),
        ("How many modes does the Philips Sonicare Series 7100 have?", "mode_details"),
        ("Does the Philips Sonicare Prestige 9900 have coverage tracking?", "coverage_tracking_details"),
    ],
)
def test_attribute_questions_reach_their_detail_records(real_engine, query, detail_attribute):
    result = answer_query(query, real_engine, no_llm=True)
    cited = {cid for seg in result.segments if seg.valid for cid in seg.citation_ids}
    assert any(f":{detail_attribute}:" in c for c in cited), (query, cited)


def test_not_stated_flag_keeps_the_exact_template_and_no_detail_segments(real_engine):
    result = answer_query("Does the Oral-B iO Series 4 Clean & Gum Care have Bluetooth?", real_engine, no_llm=True)
    assert result.answer_text.startswith("The page does not mention Bluetooth connectivity for")
    assert result.sources_text == ""


def test_what_is_the_does_not_pull_in_the_description():
    assert "description" not in detect_attributes("What is the battery life of the Philips Sonicare 4100?")
    assert "description" in detect_attributes("What is the Philips Sonicare 4100?")
    assert "description" in detect_attributes("Describe the Philips Sonicare 4100 and its battery life")


# --- the map itself -----------------------------------------------------------------------------------------

def test_every_related_attribute_name_is_a_real_column():
    import csv

    import pyarrow.parquet as pq

    from src.rag.config import COMPARISON_TABLE_CSV

    # the parquet is the RAG schema; a few evidence-backed columns (battery_life_note, brush_head_design) are CSV-only
    with COMPARISON_TABLE_CSV.open(encoding="utf-8-sig", newline="") as fh:
        csv_columns = set(next(csv.reader(fh)))
    columns = set(pq.read_schema(COMPARISON_TABLE_PARQUET).names) | csv_columns
    names = set(RELATED_ATTRIBUTES) | {v for vs in RELATED_ATTRIBUTES.values() for v in vs}
    assert names - columns == set(), names - columns
    assert names - set(ATTRIBUTE_LABELS) == set(), names - set(ATTRIBUTE_LABELS)


def test_expand_attributes_keeps_asked_first_and_has_no_duplicates():
    assert expand_attributes(["pressure_sensing"]) == ["pressure_sensing", "pressure_sensing_details"]
    out = expand_attributes(["timer", "timer_description"])
    assert out[:2] == ["timer", "timer_description"] and len(out) == len(set(out))
    assert expand_attributes(["colors"]) == ["colors"]


# --- readable evidence + sources ---------------------------------------------------------------------------------

def test_spec_json_is_rendered_as_the_line_the_shopper_sees():
    raw = ('{"csItemRank":"1","csItemName":"Operating time (full to empty)","csItemCode":"X","csValue":'
           '[{"csValueRank":"1","csValueName":"14 days","csValueCode":"Y"}]}')
    assert readable_evidence(raw) == "Operating time (full to empty): 14 days"
    assert readable_evidence("A plain sentence.") == "A plain sentence."
    truncated = '{"csItemName":"Battery type","csValue":[{"csValueName":"Lithium ION"},{"csValueName":"Li'
    assert readable_evidence(truncated) == "Battery type: Lithium ION"


def test_no_retrieval_document_shows_raw_json_to_the_user(real_engine):
    for d in real_engine.retrieval_documents:
        assert not d.shown_text.lstrip().startswith('{"cs'), d.evidence_id


def test_each_source_shows_title_section_and_a_quote_from_the_evidence(real_engine):
    result = answer_query(Q_4100, real_engine, no_llm=True)
    sources = result.sources_text
    assert "Sonicare 4100 Rechargeable toothbrush -- official product page" in sources
    assert "Section: " in sources
    by_id = {d.evidence_id: d for d in real_engine.retrieval_documents}
    cited = [cid for seg in result.segments if seg.valid for cid in seg.citation_ids]
    for cid in cited:
        assert f'Quote: "{by_id[cid].shown_text}"' in sources


def test_long_quotes_are_truncated_and_markdown_line_breaks_are_kept():
    doc = _doc("claims", "word " * 200)
    text = render_sources_section({doc.evidence_id: "S1"}, {doc.evidence_id: doc})
    quote_line = next(line for line in text.splitlines() if line.startswith("Quote:"))
    assert quote_line.rstrip().endswith('..."') and len(quote_line) < 300
    assert all(line.endswith("  ") for line in text.splitlines() if line.startswith(("[S1]", "Section:", "Quote:")))


# --- figures are quoted exactly -------------------------------------------------------------------------------------

def _validate(text, evidence_text):
    doc = _doc("claims", evidence_text)
    return validate_segment(Segment(text=text, citation_ids=[doc.evidence_id]), {doc.evidence_id: doc}, [], None, None)


def test_a_figure_must_match_as_a_whole_number():
    assert not _validate("It removes up to 5x more plaque.", "Removes up to 15x more plaque").valid
    assert not _validate("It removes 70% more plaque.", "Removes 700% more plaque").valid
    assert _validate("It removes up to 700% more plaque.", "Enjoy up to 700% more plaque removal").valid


def test_thousands_separators_and_decimals_match_exactly():
    assert _validate("It makes 62,000 brush movements per minute.", "62,000 brush movements / min").valid
    assert _validate("It makes 62000 brush movements per minute.", "62,000 brush movements / min").valid
    assert not _validate("It makes 6,200 brush movements per minute.", "62,000 brush movements / min").valid
    assert _validate("The battery lasts 3.5 hours.", "Battery lasts 3.5 hours").valid
    assert not _validate("The battery lasts 3 hours.", "Battery lasts 13 hours").valid


def test_product_name_numbers_are_not_claims():
    doc = _doc("claims", "Gently removes plaque", name="Sonicare 4100 Rechargeable toothbrush")
    seg = Segment(text="The Sonicare 4100 gently removes plaque.", citation_ids=[doc.evidence_id])
    assert validate_segment(seg, {doc.evidence_id: doc}, [], None, None).valid


def test_system_prompt_asks_for_exact_figures_and_specifics():
    from src.rag.generation import _build_system_prompt

    prompt = _build_system_prompt()
    assert "exactly as the evidence has them" in prompt and "lead with the direct answer" in prompt
    assert "bare label" in prompt


def test_fallback_drops_a_bare_label_only_when_richer_evidence_exists():
    from src.rag.generation import _fallback_segments_from_evidence

    label = _doc("pressure_sensing", "Pressure Sensor")
    detail = _doc("pressure_sensing_details", "It has a pressure sensor with haptic feedback.")
    detail.evidence_id = "p1:pressure_sensing_details:0"
    both = _fallback_segments_from_evidence([detail, label])
    assert [s.text for s in both] == ["Sonicare Test 4100: It has a pressure sensor with haptic feedback."]
    alone = _fallback_segments_from_evidence([label])
    assert [s.text for s in alone] == ["Sonicare Test 4100: Pressure Sensor"]
    json.dumps(alone[0].citation_ids)


def test_equal_numbers_in_a_different_notation_match():
    assert _validate("You save $20 and pay $69.99.", "$69.99\n$89.99\nSave $20.00").valid
    assert _validate("The list price is $89.99.", "$69.99\n$89.99\nSave $20.00").valid
    assert not _validate("You save $2 on it.", "$69.99\n$89.99\nSave $20.00").valid
    assert _validate("It runs up to 14 days.", "Operating time 14.0 days").valid


def test_system_prompt_pins_figure_notation_multiple_figures_and_product_name():
    from src.rag.generation import _build_system_prompt

    prompt = _build_system_prompt()
    assert 'write "5x" as 5x' in prompt and 'never "5 times"' in prompt
    assert "report each of them" in prompt
    assert "Name the product the question is about in your first segment" in prompt
