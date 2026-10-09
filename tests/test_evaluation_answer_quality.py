"""The answer-quality metrics (duplicate sources, redundant segments, jargon) on synthetic answers."""
from __future__ import annotations

from types import SimpleNamespace

from src.evaluation.answer_quality import (
    aggregate, duplicate_sources, jargon_hits, measure_answer, parse_sources, redundant_segments, template_jargon,
)


def _doc(eid, text, product="p1", attribute="battery_details"):
    return SimpleNamespace(evidence_id=eid, product_id=product, evidence_text=text, shown_text=text, attribute=attribute)


def _seg(text, ids, valid=True):
    return SimpleNamespace(text=text, citation_ids=ids, valid=valid)


SOURCES = (
    "Sources\n\n"
    "[S1] Sonicare Test -- official product page  \nQuote: \"It lasts 14 days.\"  \nhttps://x\n\n"
    "[S2] Sonicare Test -- official product page  \nQuote: \"It lasts 14 days.\"  \nhttps://x\n\n"
    "[S3] Sonicare Test -- official product page  \nQuote: \"It charges in 3 hours.\"  \nhttps://x\n"
)


def test_sources_are_read_back_from_the_rendered_section():
    entries = parse_sources(SOURCES)
    assert [e["marker"] for e in entries] == ["S1", "S2", "S3"]
    assert entries[0]["product"] == "Sonicare Test" and entries[0]["quote"] == "It lasts 14 days."


def test_a_repeated_quote_for_the_same_product_is_a_duplicate_source():
    assert duplicate_sources(SOURCES) == {"shown": 3, "duplicates": 1}
    other_product = SOURCES.replace("[S2] Sonicare Test", "[S2] Another Brush")
    assert duplicate_sources(other_product) == {"shown": 3, "duplicates": 0}  # same sentence, different product
    assert duplicate_sources("") == {"shown": 0, "duplicates": 0}


def test_a_segment_repeating_an_earlier_quote_and_adding_nothing_is_redundant():
    docs = {"a": _doc("a", "It lasts 14 days."), "b": _doc("b", "It lasts 14 days.", attribute="battery_life_raw")}
    segs = [_seg("Sonicare Test: It lasts 14 days.", ["a"]), _seg("It lasts 14 days.", ["b"])]
    assert redundant_segments(segs, docs) == {"shown": 2, "redundant": 1}


def test_a_second_segment_with_a_new_number_or_word_is_kept():
    docs = {"a": _doc("a", "It has 2 modes."), "b": _doc("b", "It has 2 modes.", attribute="mode_details")}
    new_number = [_seg("The brush has 2 modes.", ["a"]), _seg("The brush has 7 modes.", ["b"])]
    assert redundant_segments(new_number, docs)["redundant"] == 0
    new_word = [_seg("The brush has 2 modes.", ["a"]), _seg("The brush has 2 modes, including Whitening.", ["b"])]
    assert redundant_segments(new_word, docs)["redundant"] == 0


def test_invalid_and_uncited_segments_are_not_counted_as_shown():
    docs = {"a": _doc("a", "It lasts 14 days.")}
    segs = [_seg("x", ["a"], valid=False), _seg("The page does not mention Bluetooth.", [])]
    assert redundant_segments(segs, docs) == {"shown": 0, "redundant": 0}


def test_jargon_in_our_wording_is_flagged_but_page_quotes_are_not():
    assert jargon_hits('The page does not provide a normalized value.') == ["normalized"]
    assert jargon_hits("The available source does not mention price.") == ["the available source"]
    assert jargon_hits("That product is not in the corpus.") == ["corpus"]
    # inside quotation marks it is the page's wording, not ours
    assert jargon_hits('It says: “Check the status of each record.”') == []
    assert jargon_hits('It says: "Check the status of each record."') == []
    # the same words outside quotes are ours
    assert set(jargon_hits("Check the status of each record.")) == {"status", "record"}


def test_quote_lines_and_verbatim_evidence_sentences_are_not_scanned():
    docs = {"a": _doc("a", "The charging record shows status in the app.")}
    segs = [_seg("Sonicare Test: The charging record shows status in the app.", ["a"])]
    text = segs[0].text + "\n\nSources\n\n[S1] Sonicare Test -- official product page\nQuote: \"The charging record shows status in the app.\"\n"
    assert jargon_hits(text, segs, docs) == []
    assert jargon_hits("The charging record shows status in the app.") != []  # not quoted, not cited: ours


def test_negative_control_the_checker_can_fail():
    assert "normalized" in jargon_hits("The source does not provide a normalized value for battery life.")


def test_measure_and_aggregate_add_up():
    docs = {"a": _doc("a", "It lasts 14 days."), "b": _doc("b", "It lasts 14 days.", attribute="battery_life_raw")}
    segs = [_seg("Sonicare Test: It lasts 14 days.", ["a"]), _seg("It lasts 14 days.", ["b"])]
    answer = SimpleNamespace(answer_text="Sonicare Test: It lasts 14 days. [S1]", sources_text=SOURCES, segments=segs)
    one = measure_answer(answer, docs)
    assert one["sources_shown"] == 3 and one["duplicate_sources"] == 1 and one["redundant_segments"] == 1 and one["jargon_terms"] == []
    clean = {"sources_shown": 2, "duplicate_sources": 0, "segments_shown": 2, "redundant_segments": 0, "jargon_terms": []}
    agg = aggregate([one, clean, None])
    assert agg["cases"] == 2 and agg["duplicate_sources"] == 1 and agg["sources_shown"] == 5
    assert agg["duplicate_source_rate"] == 1 / 5 and agg["jargon_rate"] == 0.0


def test_template_inventory_is_built_from_the_real_templates():
    flagged = template_jargon()
    # whatever the templates say today, every name maps to the blocked terms found in it
    assert all(isinstance(v, list) and v for v in flagged.values())
