"""The evaluation harness checks what it claims to check: content phrases, a product-name
check that cannot pass the wrong product, per-category quality gates, and report text built
from real data."""
from __future__ import annotations

from pathlib import Path

from src.evaluation.content_checks import check_content, phrase_found, summarize
from src.evaluation.quality_gates import QualityGates, evaluate_gates
from src.evaluation.report import render_category_report, render_index
from src.evaluation.schema import load_test_suite
from src.rag.query_understanding import product_mentioned

ROOT = Path(__file__).resolve().parents[1]


# --- content checks -----------------------------------------------------------------------------------

def test_figures_match_as_whole_tokens():
    assert not phrase_found("It removes up to 15x more plaque.", "5x")
    assert phrase_found("It removes up to 5x more plaque.", "5x")
    assert phrase_found("It removes 700% more plaque.", "700%")
    assert not phrase_found("It removes 1700% more plaque.", "700%")
    assert not phrase_found("Now $169.99.", "69.99")
    assert phrase_found("Now $69.99.", "69.99")
    assert phrase_found("$69.99 after a $20 discount from $89.99.", "89.99")
    assert not phrase_found("It lasts 13.5 hours.", "3.5")


def test_normalisation_is_comparison_only():
    assert phrase_found("It has Real‑time coaching.", "real-time coaching")
    assert phrase_found("62,000 movements per minute", "62000")
    assert phrase_found("Haptic feedback", "haptic")
    assert not phrase_found("unhaptic", "haptic")  # words match on boundaries too


def test_check_content_reports_each_phrase_and_forbidden_phrases():
    checks = check_content("The 9900 gives real-time coaching.", ["real-time coaching", "SenseIQ"], ["9300"])
    assert [(c["phrase"], c["ok"]) for c in checks] == [("real-time coaching", True), ("SenseIQ", False), ("9300", True)]
    assert summarize(checks) == "real-time coaching found, SenseIQ MISSING, 9300 absent"
    bad = check_content("It is better than the 9300.", [], ["9300"])
    assert bad[0]["ok"] is False and summarize(bad) == "9300 PRESENT (forbidden)"


def test_must_include_survives_loading_the_suite():
    suite = load_test_suite(ROOT / "data" / "evaluation" / "test_cases.json")
    by_id = {c.id: c for c in suite.cases}
    assert by_id["fact_included_items_io9"].must_include == ["4", "travel case", "charger"]
    assert by_id["calc_price_diff_dc9000_vs_io9"].must_include == ["199.96", "329.99", "130.03"]
    assert by_id["partial_battery_life_days_9900_vs_io9"].must_include == ["14"]
    assert by_id["conflict_mode_count_dc9000"].must_not_include == ["Sensitive mode", "five modes are"]
    assert len(suite.cases) >= 15


# --- the relaxed product-name check cannot pass the wrong product ---------------------------------------------

CGC = ("oralb_io-series-4-clean-gum-care-rechargeable-electric-toothbrush",
       "Oral-B iO Series 4 Clean & Gum Care Rechargeable Electric Toothbrush, Azalea Pink")
PLAIN = ("oralb_io-series-4-rechargeable-electric-toothbrush",
         "Oral-B iO Series 4 Rechargeable Electric Toothbrush, Matte White")


def test_an_iO4_matte_white_answer_does_not_count_for_clean_gum_care_and_vice_versa():
    assert not product_mentioned("The iO4 Matte White has an app.", *CGC)
    assert product_mentioned("The iO4 Matte White has an app.", *PLAIN)
    assert not product_mentioned(f"The {PLAIN[1]} has an app.", *CGC)
    assert not product_mentioned("The iO4 Clean Gum Care has an app.", *PLAIN)
    assert product_mentioned("The iO4 Clean Gum Care has an app.", *CGC)


def test_a_short_correct_iO4_answer_is_accepted_when_no_other_iO4_is_named():
    assert product_mentioned("The iO4 has an app.", *CGC)
    assert product_mentioned(f"The {CGC[1]} has an app.", *CGC)


def test_unambiguous_aliases_still_count_and_other_models_do_not():
    pid, name = "philips_hx4031-71", "Sonicare 4100 Rechargeable toothbrush"
    assert product_mentioned("The Philips Sonicare 4100 has a timer.", pid, name)
    assert not product_mentioned("The Philips Sonicare 9900 has a timer.", pid, name)


# --- per-category quality gates ---------------------------------------------------------------------------

def _gates_json(tmp_path) -> str:
    path = tmp_path / "quality_gates.json"
    path.write_text(
        '{"application": {"comparison_arithmetic_pass_rate_min": 1.0}, "retriever": {"contextual_precision_min": 0.6}}',
        encoding="utf-8",
    )
    return str(path)


def test_a_category_not_in_the_report_is_skipped_not_failed(tmp_path):
    gates = QualityGates.load(_gates_json(tmp_path))
    # "retriever" absent entirely from the report (e.g. --category application only)
    results = evaluate_gates(gates, {"application": {"aggregate": {"ComparisonArithmetic": {"pass_rate": 1.0}}}})
    assert [r.name for r in results] == ["application.comparison_arithmetic_pass_rate_min"]
    assert results[0].passed


def test_an_unrun_metric_inside_a_run_category_reports_actual_none_not_a_pass(tmp_path):
    gates = QualityGates.load(_gates_json(tmp_path))
    results = evaluate_gates(gates, {"retriever": {"aggregate": {}}})
    gate = next(r for r in results if r.name == "retriever.contextual_precision_min")
    assert gate.actual is None and not gate.passed


# --- report rendering ---------------------------------------------------------------------------------------

def _minimal_report() -> dict:
    return {
        "timestamp": "t", "mode": "offline", "judge_model": "gpt-4.1", "categories_run": ["retriever"],
        "corpus": {"product_count": 21, "evidence_count": 1617},
        "retriever": {
            "judge_model": "gpt-4.1",
            "cases": [{"id": "c1", "category": "direct_attribute", "exercised": True, "reason": "ok", "retrieved_chunk_count": 1, "metrics": []}],
            "total_cases": 1, "exercised_cases": 1,
            "aggregate": {"ContextualPrecisionMetric": {"cases": 1, "exercised": 1, "mean_score": 0.9, "pass_rate": 1.0}},
        },
        "quality_gates": [{"name": "retriever.contextual_precision_min", "threshold": 0.6, "actual": 0.9, "passed": True, "comparison": "min"}],
        "quality_gates_all_passed": True,
    }


def test_index_links_only_categories_actually_run():
    text = render_index(_minimal_report())
    assert "retriever.md" in text and "application.md" not in text
    assert "**Overall: PASS**" in text


def test_category_report_shows_not_run_when_absent():
    text = render_category_report({"timestamp": "t"}, "application")
    assert "Not run this invocation." in text


def test_retriever_report_shows_the_metric_table():
    text = render_category_report(_minimal_report(), "retriever")
    assert "ContextualPrecisionMetric" in text and "0.900" in text
