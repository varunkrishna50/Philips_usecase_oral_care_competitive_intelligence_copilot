"""Concise, de-duplicated, readable answers: one source per distinct quote, no restated facts, at most two
citations per segment (never leaving a figure uncited), related facts limited to what answers the question,
and plain wording in every template."""
from __future__ import annotations

import re
from types import SimpleNamespace

from src.evaluation.answer_quality import JARGON_RE, jargon_hits, measure_answer, template_jargon, template_samples
from src.rag.answer_assembly import (
    cap_citations, directness, drop_redundant_segments, limit_related, prepare_evidence, quote_key,
)
from src.rag.citations import (
    Segment, ValidatedSegment, assign_markers, render_answer_text, render_sources_section, validate_segment,
)
from src.rag.generation import _build_system_prompt, answer_query
from src.rag.models import RetrievalDocument


def _doc(eid, text, attribute="battery_details", product="p1", name="Sonicare Test 4100", section="A section"):
    return RetrievalDocument(
        document_id=f"doc:{eid}", evidence_id=eid, product_id=product, brand="Philips", product_name=name,
        attribute=attribute, value=text, status="stated", evidence_text=text, source_url="https://www.usa.philips.com/c-p/X",
        source_section=section, source_file="f.html",
    )


SENTENCE = "The battery lasts up to 14 days on one charge."


# --- one source per distinct quote -----------------------------------------------------------------------

def test_three_records_holding_one_sentence_become_one_record():
    docs = [_doc("e1", SENTENCE, "battery_details"), _doc("e2", SENTENCE, "battery_life_raw"), _doc("e3", SENTENCE, "battery_life_days")]
    kept, aliases = prepare_evidence(docs, ["battery_life_days", "battery_life_raw"])
    assert [d.evidence_id for d in kept] == ["e3"]  # the asked attribute's own record wins
    assert aliases == {"e1": "e3", "e2": "e3"}


def test_the_same_sentence_for_different_products_is_not_a_duplicate():
    docs = [_doc("a", "Pressure Sensor", "pressure_sensing", product="p1"), _doc("b", "Pressure Sensor", "pressure_sensing", product="p2")]
    kept, _ = prepare_evidence(docs, ["pressure_sensing"])
    assert len(kept) == 2


def test_a_quote_a_template_already_shows_is_not_offered_again():
    docs = [_doc("e1", SENTENCE), _doc("e2", "It charges in 3 hours.")]
    kept, _ = prepare_evidence(docs, ["battery_life_days"], already_quoted={quote_key(docs[0])})
    assert [d.evidence_id for d in kept] == ["e2"]


def test_sources_are_numbered_per_distinct_quote_and_listed_once():
    e1, e2 = _doc("e1", SENTENCE, "battery_details"), _doc("e2", SENTENCE, "battery_life_raw")
    e3 = _doc("e3", "It charges in 3 hours.")
    by_id = {d.evidence_id: d for d in (e1, e2, e3)}
    segs = [ValidatedSegment("It lasts 14 days.", ["e1", "e2"], True, trusted=False),
            ValidatedSegment("It charges in 3 hours.", ["e3"], True, trusted=False)]
    markers = assign_markers(segs, by_id)
    assert markers == {"e1": "S1", "e2": "S1", "e3": "S2"}
    text = render_answer_text(segs, markers)
    assert "[S1] [S1]" not in text and text.count("[S1]") == 1
    sources = render_sources_section(markers, by_id)
    assert sources.count("[S1]") == 1 and sources.count("Quote:") == 2


def test_markers_without_evidence_lookup_still_work_per_id():
    segs = [ValidatedSegment("x", ["e1", "e2"], True, trusted=False)]
    assert assign_markers(segs) == {"e1": "S1", "e2": "S2"}


# --- restated facts ----------------------------------------------------------------------------------------

def _vs(text, ids, trusted=False):
    return ValidatedSegment(text, ids, True, trusted=trusted)


def test_a_second_segment_that_adds_nothing_is_dropped():
    by_id = {"a": _doc("a", SENTENCE), "b": _doc("b", SENTENCE, "battery_life_raw")}
    kept = drop_redundant_segments([_vs("The battery lasts up to 14 days.", ["a"]), _vs("The battery lasts up to 14 days.", ["b"])], by_id)
    assert len(kept) == 1


def test_a_second_segment_with_a_different_number_is_kept():
    by_id = {"a": _doc("a", "It has 2 modes.", "mode_details"), "b": _doc("b", "It has 2 modes.", "modes")}
    kept = drop_redundant_segments([_vs("The brush has 2 modes.", ["a"]), _vs("The brush has 7 modes.", ["b"])], by_id)
    assert len(kept) == 2


def test_a_second_segment_with_a_different_word_is_kept():
    by_id = {"a": _doc("a", "It has 2 modes.", "mode_details"), "b": _doc("b", "It has 2 modes.", "modes")}
    kept = drop_redundant_segments([_vs("The brush has 2 modes.", ["a"]), _vs("The brush has 2 modes, including Whitening.", ["b"])], by_id)
    assert len(kept) == 2


def test_trusted_and_invalid_segments_are_never_dropped():
    by_id = {"a": _doc("a", SENTENCE)}
    trusted = _vs("The battery lasts up to 14 days.", ["a"], trusted=True)
    invalid = ValidatedSegment("junk", [], False, ["x"], trusted=False)
    kept = drop_redundant_segments([_vs("The battery lasts up to 14 days.", ["a"]), trusted, invalid], by_id)
    assert kept == [kept[0], trusted, invalid]


# --- citation cap ------------------------------------------------------------------------------------------

def _cap_setup():
    docs = {
        "own": _doc("own", "The battery lasts up to 14 days.", "battery_life_days"),
        "detail": _doc("detail", "Experience up to 14 days of use.", "battery_details"),
        "marketing": _doc("marketing", "A great brushing experience.", "claims"),
        "hours": _doc("hours", "Charging takes 3 hours.", "charge_time_hours"),
        "pct": _doc("pct", "Removes 700% more plaque.", "claims"),
    }
    return docs


def test_the_cap_keeps_the_two_most_direct_citations():
    docs = _cap_setup()
    seg = Segment("It lasts 14 days.", ["marketing", "detail", "own"], trusted=False)
    out = cap_citations(seg, docs, ["battery_life_days"])
    assert out.citation_ids == ["detail", "own"] or set(out.citation_ids) == {"own", "detail"}
    assert "marketing" not in out.citation_ids


def test_two_figures_from_two_quotes_keep_both_quotes():
    docs = _cap_setup()
    seg = Segment("It lasts 14 days and charges in 3 hours.", ["own", "detail", "marketing", "hours"], trusted=False)
    out = cap_citations(seg, docs, ["battery_life_days"])
    assert "own" in out.citation_ids or "detail" in out.citation_ids
    assert "hours" in out.citation_ids  # the only quote backing "3"
    by_id = {d.evidence_id: d for d in docs.values()}
    assert validate_segment(out, by_id, [], None, None).valid


def test_three_figures_that_need_three_quotes_keep_three():
    docs = _cap_setup()
    seg = Segment("14 days, 3 hours and 700% more plaque.", ["own", "hours", "pct", "marketing"], trusted=False)
    out = cap_citations(seg, docs, ["battery_life_days"])
    assert {"own", "hours", "pct"} <= set(out.citation_ids) and "marketing" not in out.citation_ids
    assert validate_segment(out, {d.evidence_id: d for d in docs.values()}, [], None, None).valid


def test_the_validator_still_passes_after_redundant_quotes_are_dropped():
    docs = _cap_setup()
    seg = Segment("It lasts up to 14 days.", ["own", "detail", "marketing"], trusted=False)
    before = validate_segment(seg, {d.evidence_id: d for d in docs.values()}, [], None, None)
    out = cap_citations(seg, docs, ["battery_life_days"])
    after = validate_segment(out, {d.evidence_id: d for d in docs.values()}, [], None, None)
    assert before.valid and after.valid and len(out.citation_ids) == 2


def test_directness_orders_own_then_detail_then_other():
    own, detail, other = _doc("a", "x", "pressure_sensing"), _doc("b", "y", "pressure_sensing_details"), _doc("c", "z", "claims")
    asked = ["pressure_sensing"]
    assert directness(own, asked) < directness(detail, asked) < directness(other, asked)


# --- related facts nobody asked for ---------------------------------------------------------------------------

def test_topic_pure_details_are_all_kept_and_mixed_ones_are_limited():
    pure = [_doc("p1", "Haptic feedback.", "pressure_sensing_details"), _doc("p2", "It vibrates.", "pressure_sensing_details")]
    assert len(limit_related(pure, set(), scores={})) == 2
    mixed = [_doc("m1", "Charging takes 3 hours.", "battery_details"), _doc("m2", "Travel case included.", "battery_details"),
             _doc("m3", "Display shows battery.", "battery_details")]
    kept = limit_related(mixed, set(), scores={"m1": 0.87, "m2": 0.74, "m3": 0.75, "top": 0.94})
    assert kept == []  # none within 95% of the best hit


def test_a_mixed_record_close_to_the_best_hit_is_kept_but_only_one():
    mixed = [_doc("m1", "Up to 14 days of use.", "battery_details"), _doc("m2", "Indicator.", "battery_details")]
    kept = limit_related(mixed, set(), scores={"m1": 0.97, "m2": 0.96, "top": 0.98})
    assert [d.evidence_id for d in kept] == ["m1"]


def test_a_mixed_record_the_question_names_is_kept():
    mixed = [_doc("c1", "Charging takes 3 hours.", "charging_details")]
    kept = limit_related(mixed, {"charging_details"}, scores={})
    assert [d.evidence_id for d in kept] == ["c1"]


# --- plain wording ------------------------------------------------------------------------------------------------

def test_no_template_we_write_contains_internal_wording():
    assert template_samples()
    assert template_jargon() == {}


def test_every_offline_answer_is_free_of_internal_wording(real_engine):
    from src.evaluation.schema import load_test_suite
    from pathlib import Path

    suite = load_test_suite(Path(__file__).resolve().parents[1] / "data" / "evaluation" / "test_cases.json")
    by_id = {d.evidence_id: d for d in real_engine.retrieval_documents}
    bad = {}
    for case in suite.cases:
        if case.category == "prompt_injection_evidence":
            continue
        answer = answer_query(case.question, real_engine, no_llm=True)
        hits = measure_answer(answer, by_id)["jargon_terms"]
        if hits:
            bad[case.id] = hits
    assert bad == {}, bad


def test_page_wording_inside_quotes_is_not_jargon_but_our_wording_is():
    assert jargon_hits("It says: “Check the status of each record.”") == []
    assert jargon_hits("Check the status of each record.") != []
    assert "normalized" in jargon_hits("The page does not provide a normalized value.")  # negative control


def test_the_prompt_stays_general_and_asks_for_no_repeats():
    prompt = _build_system_prompt()
    for specific in ("iO10", "iO Series", "battery", "Sonicare 4100", "charge time", "travel case"):
        assert specific.lower() not in prompt.lower().replace("sonicare and oral-b", ""), specific
    assert "State each fact once" in prompt and "Use only facts that answer the question" in prompt
    assert "Write in plain words" in prompt


# --- the iO10 battery question -------------------------------------------------------------------------------------

IO10_Q = "What is the battery life of the iO Series 10 Rechargeable Electric Toothbrush, Cosmic Black?"


def test_io10_battery_answer_is_one_plain_sentence_with_one_source(real_engine):
    answer = answer_query(IO10_Q, real_engine, no_llm=True)
    assert "The page does not state how many days the battery lasts" in answer.answer_text
    assert "normalized" not in answer.answer_text.lower()
    assert answer.sources_text.count("Quote:") == 1  # the one sentence the page gives, shown once
    for unrelated in ("3 hours", "travel case", "display", "indicator"):
        assert unrelated not in answer.answer_text.lower(), unrelated
    assert not re.search(JARGON_RE, answer.answer_text.replace("“", '"').split('"')[0])


def test_pressure_answer_keeps_both_detail_sentences(real_engine):
    answer = answer_query("What pressure-sensing feature does the Philips Sonicare 4100 have?", real_engine, no_llm=True)
    assert "haptic feedback" in answer.answer_text and "gentle vibrations" in answer.answer_text
    assert answer.sources_text.count("Quote:") == 2
