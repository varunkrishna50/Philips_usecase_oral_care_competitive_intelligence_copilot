"""The evaluation harness checks what it claims to check: content phrases, uncited-segment
breakdown, report sentences built from real counts, a negative control for the groundedness
checker, and a product-name check that cannot pass the wrong product."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation import runner
from src.evaluation.citation_metrics import aggregate_citation_metrics, classify_uncited, compute_citation_metrics
from src.evaluation.content_checks import check_content, phrase_found, summarize
from src.evaluation.gold_labels import load_gold_corpus
from src.evaluation.groundedness import run_negative_controls
from src.evaluation.quality_gates import QualityGates, evaluate_gates
from src.evaluation.report import render_report
from src.evaluation.schema import TestCase, load_test_suite
from src.rag.citations import ValidatedSegment
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


def _case(**kw) -> TestCase:
    base = dict(id="c1", question="q", category="direct_attribute", expected_behavior="x", expected_answer_type="fact")
    return TestCase(**{**base, **kw})


def test_runner_fails_a_case_whose_answer_misses_a_required_phrase(monkeypatch):
    ok = runner.CaseResult("c1", "direct_attribute", "q", True, "pass", "product/attribute mentioned with citations",
                           answer_text="It has haptic feedback.")
    monkeypatch.setattr(runner, "_evaluate_test_case", lambda *a, **k: ok)
    passed = runner.evaluate_test_case(_case(must_include=["haptic"]), None, None, "offline", None)
    assert passed.verdict == "pass" and passed.content_checks[0]["found"]

    ok2 = runner.CaseResult("c1", "direct_attribute", "q", True, "pass", "d", answer_text="It has a pressure sensor.")
    monkeypatch.setattr(runner, "_evaluate_test_case", lambda *a, **k: ok2)
    failed = runner.evaluate_test_case(_case(must_include=["haptic"]), None, None, "offline", None)
    assert failed.verdict == "fail" and failed.passed is False and "haptic MISSING" in failed.detail

    ok3 = runner.CaseResult("c1", "direct_attribute", "q", True, "pass", "d", answer_text="Costs $399.99.")
    monkeypatch.setattr(runner, "_evaluate_test_case", lambda *a, **k: ok3)
    forbidden = runner.evaluate_test_case(_case(must_not_include=["399.99"]), None, None, "offline", None)
    assert forbidden.verdict == "fail"


def test_a_case_without_phrases_is_untouched(monkeypatch):
    res = runner.CaseResult("c1", "direct_attribute", "q", True, "pass", "d", answer_text="anything")
    monkeypatch.setattr(runner, "_evaluate_test_case", lambda *a, **k: res)
    out = runner.evaluate_test_case(_case(), None, None, "offline", None)
    assert out.verdict == "pass" and out.content_checks == [] and out.detail == "d"


def test_must_include_survives_loading_the_suite():
    suite = load_test_suite(ROOT / "data" / "evaluation" / "test_cases.json")
    by_id = {c.id: c for c in suite.cases}
    assert by_id["attr_pressure_depth_027"].must_include == ["haptic", "gentle vibrations"]
    assert by_id["claims_conflict_029"].must_include == ["700%", "5x"]
    assert by_id["price_discount_034"].must_include == ["89.99", "69.99"]
    assert by_id["price_hidden_033"].must_not_include == ["399.99"]
    s035 = by_id["alias_sonicare9900_035"]
    assert s035.must_include == ["real-time coaching"] and s035.must_not_include == ["9300"]
    assert "variant_alias_036" in by_id and len(suite.cases) == 36


# --- citation presence ------------------------------------------------------------------------------------

def _seg(text, trusted, valid, cids):
    return ValidatedSegment(text=text, citation_ids=cids, valid=valid, trusted=trusted)


def test_uncited_segments_are_split_by_kind_and_add_up():
    segs = [
        _seg("The page does not mention Bluetooth connectivity for X.", True, True, []),
        _seg("It has a timer.", False, False, []),            # LLM segment the validator rejected
        _seg("It has a pressure sensor.", False, True, []),    # shown, factual, no citation = a real gap
        _seg("It costs $10.", False, True, ["e1"]),
        _seg("Haptic feedback.", False, True, ["e2"]),
    ]
    assert [classify_uncited(s) for s in segs] == ["absence_disclosure", "rejected_by_validator", "uncited_other", "cited", "cited"]
    m = compute_citation_metrics(segs)
    uncited = m.total_segments - round(m.presence_rate * m.total_segments)
    assert uncited == m.uncited_absence_disclosures + m.uncited_rejected_by_validator + m.uncited_other == 3
    assert (m.uncited_absence_disclosures, m.uncited_rejected_by_validator, m.uncited_other) == (1, 1, 1)
    assert m.shown_factual_segments == 3 and m.presence_rate_shown_factual == pytest.approx(2 / 3)
    agg = aggregate_citation_metrics([m, m])
    assert agg.uncited_other == 2 and agg.total_segments == 10


def test_disclosures_alone_do_not_trip_the_uncited_factual_gate():
    segs = [_seg("The page does not mention price for Y.", True, True, [])]
    m = compute_citation_metrics(segs)
    assert m.uncited_other == 0 and m.presence_rate == 0.0 and m.presence_rate_shown_factual == 1.0
    gates = evaluate_gates(QualityGates.load(), {"safety_pass_rate": 1.0, "unsupported_claim_rate": 0.0,
                                                 "exhaustive_completeness": 1.0, "uncited_factual_segments": m.uncited_other})
    assert all(g.passed for g in gates) and gates[-1].name == "uncited_factual_segments"
    failing = evaluate_gates(QualityGates.load(), {"safety_pass_rate": 1.0, "unsupported_claim_rate": 0.0,
                                                   "exhaustive_completeness": 1.0, "uncited_factual_segments": 1})
    assert not failing[-1].passed


# --- negative control ---------------------------------------------------------------------------------------

def test_groundedness_checker_flags_deliberately_false_sentences_and_passes_true_ones():
    result = run_negative_controls(load_gold_corpus())
    assert result["false_sentences"] >= 5 and result["false_flagged"] == result["false_sentences"]
    assert result["true_sentences"] >= 2 and result["true_clean"] == result["true_sentences"]
    assert result["all_ok"]
    labels = {i["label"] for i in result["items"]}
    assert {"made-up mode count", "made-up price", "made-up percentage",
            "denies a feature the product has", "names a product outside the question's scope"} <= labels


# --- the report text ----------------------------------------------------------------------------------------

def _results(product_count=21, evidence_count=1617, stated=1617):
    cm = {"total_segments": 10, "trusted_segments": 3, "untrusted_segments": 7, "presence_rate": 0.8,
          "validity_rate_all": 1.0, "validity_rate_untrusted_only": 1.0, "uncited_absence_disclosures": 1,
          "uncited_rejected_by_validator": 1, "uncited_other": 0, "presence_rate_shown_factual": 1.0,
          "shown_factual_segments": 8}
    case = {"id": "attr_pressure_depth_027", "category": "direct_attribute", "question": "q", "verdict": "pass",
            "passed": True, "detail": "d", "groundedness_issues": [],
            "content_checks": check_content("haptic and gentle vibrations", ["haptic", "gentle vibrations"], [])}
    return {
        "timestamp": "t", "mode": "offline",
        "corpus": {"product_count": product_count, "evidence_count": evidence_count, "retrieval_documents": 1617,
                   "retrieval_documents_sha256": "abc123def456", "retrieval_documents_modified": "2026-10-10T00:00:00+00:00"},
        "model_configuration": {"provider": "openai", "model": "m", "temperature": 0},
        "aggregate_metrics": {"passed_cases": 1, "exercised_cases": 1, "not_exercised_cases": 0, "total_cases": 1,
                              "safety_pass_rate": 1.0, "exhaustive_completeness": 1.0, "unsupported_claim_rate": 0.0,
                              "unsupported_claim_count": 0, "safety_not_exercised": []},
        "citation_metrics": cm,
        "retrieval_metrics": {"ablation": {"status": "skipped", "reason": "x"}, "behavioral_checks": {}},
        "status_coverage": {"stated": {"has_real_example": True, "count": stated},
                            "not_mentioned": {"has_real_example": True, "count": 5},
                            "stated_absent": {"has_real_example": False, "count": 0},
                            "ambiguous": {"has_real_example": False, "count": 0},
                            "stated_with_null_value": {"has_real_example": True, "count": 2}},
        "test_cases": [case], "quality_gates": [], "quality_gates_all_passed": True,
        "groundedness_negative_control": run_negative_controls(load_gold_corpus()),
    }


def test_report_sentences_come_from_the_real_counts():
    text = render_report(_results())
    assert "24-product" not in text and "486" not in text
    assert "21-product corpus" in text and "all 1617 evidence records carry status=`stated`" in text
    other = render_report(_results(product_count=7, evidence_count=900, stated=850))
    assert "7-product corpus" in other and "850 of 900 evidence records carry status=`stated`" in other


def test_report_shows_citation_breakdown_content_checks_negative_control_and_corpus_fingerprint():
    text = render_report(_results())
    assert "1 not-mentioned disclosure(s)" in text and "1 rejected by the citation validator" in text
    assert "**0 factual segment(s) shown without a citation**" in text
    assert "attr_pressure_depth_027`: ok -- haptic found, gentle vibrations found" in text
    assert "Checker self-test (negative control): 5/5 deliberately false sentences flagged" in text
    assert "sha256 `abc123def456`" in text and "built 2026-10-10T00:00:00+00:00" in text
    assert "reused" not in text


def test_the_committed_generated_reports_have_no_stale_counts():
    for name in ("evaluation_report.md", "evaluation_report_live.md"):
        path = ROOT / "data" / "evaluation" / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            assert "24-product" not in text and "all 486" not in text, name


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


# --- exhaustive completeness is judged by citations, not by the spelling of a product name ----------------

def test_exhaustive_completeness_counts_cited_products_even_if_the_name_is_shortened():
    from types import SimpleNamespace

    corpus = load_gold_corpus()
    attribute = "pressure_sensing"
    case = _case(id="ex", category="exhaustive", gold_attributes=[attribute], expected_answer_type="exhaustive_list")
    from src.evaluation import exhaustive_eval

    expected = exhaustive_eval.gold_exhaustive_expected(corpus, attribute, True)
    ev_of = {}
    for e in corpus.evidence:
        if e.product_id in expected and e.attribute == attribute:
            ev_of.setdefault(e.product_id, e.evidence_id)
    matches = [SimpleNamespace(product=SimpleNamespace(product_id=pid),
                               evidence=[SimpleNamespace(evidence_id=eid)]) for pid, eid in ev_of.items()]
    rr = SimpleNamespace(exhaustive_matches=matches)

    def answer(segments):
        return SimpleNamespace(retrieval_result=rr, segments=segments)

    full = [ValidatedSegment(text="It has a pressure sensor.", citation_ids=[eid], valid=True, trusted=False)
            for eid in ev_of.values()]  # no product name at all in the text
    ok, detail, _ = runner._evaluate_exhaustive_case(case, answer(full), corpus)
    assert ok, detail
    ok, detail, _ = runner._evaluate_exhaustive_case(case, answer(full[:-1]), corpus)
    assert not ok and "missing" in detail
    rejected = full[:-1] + [ValidatedSegment(text="x", citation_ids=[], valid=False, trusted=False)]
    ok, _, _ = runner._evaluate_exhaustive_case(case, answer(rejected), corpus)
    assert not ok
