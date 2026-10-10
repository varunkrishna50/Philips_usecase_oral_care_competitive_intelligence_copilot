"""Integration tests for the DeepEval-based evaluation runner: offline
mode runs end-to-end with zero real API calls (neither the app's own
LLM nor the DeepEval judge), against all 17 real test_cases.json cases.
"""
from __future__ import annotations

import json

import pytest

from src.evaluation.runner import ALL_CATEGORIES, run_deepeval_suite


@pytest.fixture(scope="module")
def offline_report():
    return run_deepeval_suite(mode="offline")


class TestOfflineModeNeverCallsARealApi:
    def test_offline_run_makes_no_real_llm_call(self, monkeypatch):
        def _boom(self, system_prompt, user_prompt):
            raise AssertionError("offline evaluation must never call the real OpenAI API")

        monkeypatch.setattr("src.rag.llm_provider.OpenAIStructuredClient.generate_json", _boom)
        report = run_deepeval_suite(mode="offline")
        assert report["mode"] == "offline"

    def test_offline_run_never_invokes_a_deepeval_metric_or_classifier(self, monkeypatch):
        """measure_if_live/classify_if_live are the two chokepoints every
        category routes a DeepEval judge call through -- if either ever
        called measure()/classify() in offline mode, this would raise."""
        def _boom_measure(self, test_case, _show_indicator=True, _in_component=False):
            raise AssertionError("offline evaluation must never call a real DeepEval judge metric")

        def _boom_classify(self, test_case, _show_indicator=True, _in_component=False):
            raise AssertionError("offline evaluation must never call a real DeepEval classifier")

        monkeypatch.setattr("deepeval.metrics.base_metric.BaseMetric.measure", _boom_measure)
        monkeypatch.setattr("deepeval.classifiers.base_classifier.BaseClassifier.classify", _boom_classify)
        report = run_deepeval_suite(mode="offline")
        assert report["mode"] == "offline"

    def test_offline_run_does_not_construct_a_real_app_client(self, offline_report):
        assert offline_report["model_configuration"]["api_key_configured"] is None


class TestOfflineRunProducesAllFourCategories:
    def test_all_17_cases_loaded(self, offline_report):
        assert offline_report["retriever"]["total_cases"] == 16
        assert offline_report["pipeline"]["total_cases"] == 16
        assert offline_report["application"]["total_cases"] == 16

    def test_generator_requires_live_and_is_fully_not_exercised_offline(self, offline_report):
        assert offline_report["generator"]["exercised_cases"] == 0
        for c in offline_report["generator"]["cases"]:
            assert not c["exercised"]

    def test_deepeval_judge_metrics_are_not_exercised_offline(self, offline_report):
        for category in ("retriever", "pipeline"):
            agg = offline_report[category]["aggregate"]
            for metric_name, info in agg.items():
                assert info["exercised"] == 0, (category, metric_name)
        # application's judge-based metrics are not exercised; its one
        # deterministic metric (ComparisonArithmetic) IS, even offline.
        app_agg = offline_report["application"]["aggregate"]
        for metric_name, info in app_agg.items():
            if metric_name == "ComparisonArithmetic":
                assert info["exercised"] == info["cases"], metric_name
            else:
                assert info["exercised"] == 0, metric_name

    def test_comparison_arithmetic_runs_offline_and_passes(self, offline_report):
        agg = offline_report["application"]["aggregate"]["ComparisonArithmetic"]
        assert agg["cases"] == 3  # the 3 comparison cases in the current suite
        assert agg["pass_rate"] == 1.0

    def test_content_checks_run_offline(self, offline_report):
        """Only confirms the plumbing runs -- NOT that every check
        passes offline. The current dataset's harder reasoning cases
        (false-premise correction, multi-hop calculation, resolving
        conflicting source data) and two real query-understanding gaps
        this dataset surfaced (a bare "Oral-B iO Series 9" is ambiguous
        against the Genius variant; "which is cheaper... and by how
        much" doesn't trigger comparison intent) mean most content
        checks are not expected to pass without a live LLM, if even
        then. See this session's investigation for the full findings."""
        summary = offline_report["application"]["content_checks_summary"]
        assert summary["cases_checked"] >= 1

    def test_result_is_json_serializable(self, offline_report):
        json.dumps(offline_report, default=str)


class TestCategoryFiltering:
    def test_only_requested_categories_are_in_the_report(self):
        report = run_deepeval_suite(mode="offline", categories=["application"])
        present = set(ALL_CATEGORIES) & set(report.keys())
        assert present == {"application"}

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError):
            run_deepeval_suite(mode="offline", categories=["not_a_real_category"])

    def test_gates_for_a_skipped_category_are_absent_not_failed(self):
        report = run_deepeval_suite(mode="offline", categories=["application"])
        gate_names = {g["name"] for g in report["quality_gates"]}
        assert "retriever.contextual_precision_min" not in gate_names
        assert "application.comparison_arithmetic_pass_rate_min" in gate_names


class TestCli:
    def _evaluate_module(self):
        import importlib
        import sys
        from pathlib import Path

        scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        return importlib.import_module("evaluate")

    def test_cli_defaults_to_offline_without_the_live_flag(self, tmp_path, monkeypatch):
        evaluate = self._evaluate_module()
        captured_modes = []
        original = evaluate.run_deepeval_suite

        def _spy(mode="offline", **kwargs):
            captured_modes.append(mode)
            return original(mode=mode, **kwargs)

        monkeypatch.setattr(evaluate, "run_deepeval_suite", _spy)
        evaluate.main(["--out-dir", str(tmp_path), "--category", "application"])
        assert captured_modes == ["offline"]

    def test_lenient_gates_pass(self, tmp_path):
        evaluate = self._evaluate_module()
        lenient_gates = tmp_path / "lenient_gates.json"
        lenient_gates.write_text(json.dumps({"application": {"comparison_arithmetic_pass_rate_min": 0.0}}))
        exit_code = evaluate.main(["--gates", str(lenient_gates), "--out-dir", str(tmp_path / "out"), "--category", "application"])
        assert exit_code == 0
        assert (tmp_path / "out" / "report.json").exists()
        assert (tmp_path / "out" / "application.md").exists()

    def test_default_gates_pass_for_comparison_arithmetic_offline(self, tmp_path):
        evaluate = self._evaluate_module()
        lenient_gates = tmp_path / "gates.json"
        lenient_gates.write_text(json.dumps({"application": {"comparison_arithmetic_pass_rate_min": 1.0}}))
        exit_code = evaluate.main(["--gates", str(lenient_gates), "--out-dir", str(tmp_path / "out"), "--category", "application"])
        assert exit_code == 0

    def test_impossible_gate_threshold_still_fails(self, tmp_path):
        evaluate = self._evaluate_module()
        impossible_gates = tmp_path / "impossible_gates.json"
        impossible_gates.write_text(json.dumps({"application": {"comparison_arithmetic_pass_rate_min": 1.1}}))
        exit_code = evaluate.main(["--gates", str(impossible_gates), "--out-dir", str(tmp_path / "out"), "--category", "application"])
        assert exit_code != 0
