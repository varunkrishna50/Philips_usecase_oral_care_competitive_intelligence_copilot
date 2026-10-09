"""Product recognition through product_aliases.csv (the fallback when no catalog name matches)."""
from __future__ import annotations

import pytest

from src.rag.generation import answer_query
from src.rag.models import ProductRecord
from src.rag.query_understanding import detect_comparison_products, detect_products


def _match(real_engine, query):
    m = detect_products(query, real_engine.product_corpus)
    return m.kind, sorted(p.product_id for p in m.candidates)


@pytest.mark.parametrize(
    "query, product_id",
    [
        ("Does the Philips Sonicare 9900 have an app?", "philips_hx9990-13"),
        ("What is the battery life of the HX9990?", "philips_hx9990-13"),
        ("How many modes does the Sonicare 9300 have?", "philips_hx9903-25"),
        ("Tell me about the Sonicare 3100 series", "philips_hx3681-03"),
        ("Is there a timer on the Philips One?", "philips_hy1200-25"),
        ("Does One for Kids have a timer?", "philips_hy1130-02"),
        ("How many modes does the iO6 have?", "oralb_io-series-6-electric-toothbrush"),
        ("What is the price of the Oral-B iO10?", "oralb_io-series-10-rechargeable-electric-toothbrush"),
    ],
)
def test_names_people_type_resolve_to_one_product(real_engine, query, product_id):
    assert _match(real_engine, query) == ("matched", [product_id])


def test_bare_io4_is_ambiguous_between_the_two_pages_not_a_guess(real_engine):
    kind, ids = _match(real_engine, "Does the iO4 have Bluetooth?")
    assert kind == "ambiguous" and len(ids) == 2 and all("series-4" in i for i in ids)
    result = answer_query("Does the iO4 have Bluetooth?", real_engine, no_llm=True)
    assert "multiple" in result.answer_text.lower() and "Please specify" in result.answer_text


def test_the_variant_alias_picks_the_right_io4(real_engine):
    assert _match(real_engine, "Does the iO4 Clean Gum Care have an app?") == (
        "matched", ["oralb_io-series-4-clean-gum-care-rechargeable-electric-toothbrush"])


def test_exact_name_matches_are_unchanged(real_engine):
    assert _match(real_engine, "What does the Philips Sonicare 4100 have?")[0] == "matched"
    assert _match(real_engine, "Philips Sonicare Prestige 9900 battery")[1] == ["philips_hx9990-13"]


def test_unknown_models_and_competitors_stay_not_in_corpus(real_engine):
    assert detect_products("What is the price of the Sonicare 1234?", real_engine.product_corpus).kind == "not_in_corpus"
    assert detect_products("How does the Colgate hum compare?", real_engine.product_corpus).kind == "not_in_corpus"


def test_category_questions_are_still_no_product(real_engine):
    assert detect_products("Which products have pressure sensing?", real_engine.product_corpus).kind == "none"
    assert detect_products("Which electric toothbrush is best?", real_engine.product_corpus).kind == "none"


def test_comparison_of_two_alias_named_products(real_engine):
    matches = detect_comparison_products("Compare the Sonicare 9900 and the iO6", real_engine.product_corpus)
    assert sorted(m.candidates[0].product_id for m in matches) == [
        "oralb_io-series-6-electric-toothbrush", "philips_hx9990-13"]


def test_aliases_of_products_not_in_the_list_are_ignored():
    fake = ProductRecord("x_1", "Philips", "Sonicare Fake 5000", "u", "u", "u", "2026-01-01")
    assert detect_products("Does the Sonicare 9900 have an app?", [fake]).kind != "matched"


def test_the_screenshot_style_question_now_gets_a_scoped_answer(real_engine):
    result = answer_query("Does the Philips Sonicare 9900 have an app and what does it do?", real_engine, no_llm=True)
    assert result.retrieval_result.product_match.kind == "matched"
    assert "Prestige 9900" in result.answer_text and "DiamondClean 9000" not in result.answer_text
    cited = {c for s in result.segments if s.valid for c in s.citation_ids}
    assert any(":app_connectivity_details:" in c for c in cited)


def test_rechargeable_in_a_product_name_is_not_a_battery_or_charging_question():
    from src.rag.query_understanding import detect_attributes

    q = "What is the price of the iO Series 10 Rechargeable Electric Toothbrush, Cosmic Black?"
    assert detect_attributes(q) == ["price"]
    assert detect_attributes("What does the Sonicare 4100 Rechargeable toothbrush cost?") == ["price"]
    # a real question about charging still counts
    assert "charging_method" in detect_attributes("How is the Sonicare 4100 charged? Is it rechargeable?")
    assert "battery_life_days" in detect_attributes("Is the toothbrush rechargeable?")


def test_product_mentioned_accepts_catalog_name_and_aliases(real_engine):
    from src.rag.query_understanding import product_mentioned

    pid = "philips_hx4031-71"
    name = next(p.product_name for p in real_engine.product_corpus if p.product_id == pid)
    assert product_mentioned(f"The {name} has a timer.", pid, name)
    assert product_mentioned("The Philips Sonicare 4100 has a timer.", pid, name)
    assert not product_mentioned("The Philips Sonicare 9900 has a timer.", pid, name)
