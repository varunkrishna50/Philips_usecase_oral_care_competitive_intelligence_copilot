"""Integration tests for the 3C.1 evaluation runner: offline mode runs
end-to-end with zero real API calls. The numeric exhaustive condition
bug this harness originally found (condition_value hardcoded to True
for every attribute) has since been fixed in src/rag/query_understanding.py
and src/rag/retrieve.py -- see tests/test_rag_query_understanding.py and
tests/test_rag_retrieve_integration.py for the production-code
regression tests. The tests below now confirm the previously-failing
case passes, proving the fix took effect end-to-end through this same
harness that originally caught it.
"""
from __future__ import annotations

import json

import pytest

from src.evaluation.runner import run_evaluation


@pytest.fixture(scope="module")
def offline_results():
    return run_evaluation(mode="offline")


class TestOfflineModeNeverCallsRealApi:
    def test_offline_run_makes_no_real_llm_call(self, monkeypatch):
        """If anything in the offline path tried to reach the real
        OpenAI API, this would raise instead of silently succeeding --
        mirrors the same technique used in test_rag_generation.py's
        comparison-never-calls-LLM test."""
        def _boom(self, system_prompt, user_prompt):
            raise AssertionError("offline evaluation must never call the real OpenAI API")

        monkeypatch.setattr("src.rag.llm_provider.OpenAIStructuredClient.generate_json", _boom)
        results = run_evaluation(mode="offline")
        assert results["mode"] == "offline"

    def test_offline_run_does_not_construct_a_real_client(self, offline_results):
        # model_configuration.api_key_configured is explicitly None in
        # offline mode (never checked/reported, since no client is used).
        assert offline_results["model_configuration"]["api_key_configured"] is None


class TestOfflineRunProducesExpectedShape:
    def test_runs_all_36_cases(self, offline_results):
        assert offline_results["aggregate_metrics"]["total_cases"] == 36

    def test_poisoned_evidence_case_is_not_exercised_offline(self, offline_results):
        case = next(c for c in offline_results["test_cases"] if c["id"] == "injection_evidence_026")
        assert case["verdict"] == "not_exercised"
        assert case["passed"] is None

    def test_numeric_exhaustive_case_now_passes_after_the_fix(self, offline_results):
        """This case (exhaustive_modes7_014) was the real bug 3C.1
        originally found and reported as a failure, per the integrity
        rule of never hiding or excluding a genuine failure. The
        production fix (detect_exhaustive_condition_value in
        src/rag/query_understanding.py) has since corrected the
        underlying behavior, so re-running the SAME unmodified test
        case through the SAME unmodified evaluation harness and gold
        labels must now show it passing -- confirming the fix, not the
        test."""
        case = next(c for c in offline_results["test_cases"] if c["id"] == "exhaustive_modes7_014")
        assert case["verdict"] == "pass", case["detail"]

    def test_other_exhaustive_cases_pass(self, offline_results):
        for case_id in ("exhaustive_pressure_012", "exhaustive_app_013"):
            case = next(c for c in offline_results["test_cases"] if c["id"] == case_id)
            assert case["verdict"] == "pass", case["detail"]

    def test_safety_cases_all_pass_or_not_exercised(self, offline_results):
        from src.evaluation.runner import SAFETY_CATEGORIES
        for case in offline_results["test_cases"]:
            if case["category"] in SAFETY_CATEGORIES:
                assert case["verdict"] in ("pass", "not_exercised"), (case["id"], case["detail"])

    def test_result_is_json_serializable(self, offline_results):
        json.dumps(offline_results, default=str)

    def test_status_coverage_reports_gaps_honestly(self, offline_results):
        coverage = offline_results["status_coverage"]
        assert coverage["stated_absent"]["has_real_example"] is False
        assert coverage["ambiguous"]["has_real_example"] is False


class TestLiveModeRequiresExplicitFlag:
    def test_cli_defaults_to_offline_without_the_live_flag(self, tmp_path, monkeypatch):
        import importlib
        import sys
        from pathlib import Path

        scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        evaluate = importlib.import_module("evaluate")

        captured_modes = []
        original = evaluate.run_evaluation

        def _spy(mode="offline", **kwargs):
            captured_modes.append(mode)
            return original(mode=mode, **kwargs)

        monkeypatch.setattr(evaluate, "run_evaluation", _spy)
        out_path = tmp_path / "results.json"
        report_path = tmp_path / "report.md"
        evaluate.main(["--out", str(out_path), "--report-out", str(report_path)])
        assert captured_modes == ["offline"]


class TestQualityGateCliExitBehavior:
    def test_lenient_gates_pass(self, tmp_path):
        import importlib
        import sys
        from pathlib import Path

        scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        evaluate = importlib.import_module("evaluate")

        lenient_gates = tmp_path / "lenient_gates.json"
        lenient_gates.write_text(json.dumps({
            "safety_pass_rate_min": 0.0, "unsupported_claim_rate_max": 1.0, "exhaustive_completeness_min": 0.0,
        }))
        out_path = tmp_path / "results.json"
        report_path = tmp_path / "report.md"
        exit_code = evaluate.main([
            "--gates", str(lenient_gates), "--out", str(out_path), "--report-out", str(report_path),
        ])
        assert exit_code == 0
        assert out_path.exists()
        assert report_path.exists()

    def test_default_gates_now_pass_after_the_exhaustive_fix(self, tmp_path):
        """Previously this asserted exit_code != 0, because the
        now-fixed numeric-exhaustive bug dragged exhaustive_completeness
        below its gate threshold. With the production bug fixed, the
        same strict default gates must now pass -- confirming the fix
        end-to-end through the real CLI, not just the runner function."""
        import importlib
        import sys
        from pathlib import Path

        scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        evaluate = importlib.import_module("evaluate")

        out_path = tmp_path / "results.json"
        report_path = tmp_path / "report.md"
        exit_code = evaluate.main(["--out", str(out_path), "--report-out", str(report_path)])
        assert exit_code == 0

    def test_impossible_gate_threshold_still_fails(self, tmp_path):
        """Keeps the FAIL exit path itself covered, independent of
        whatever bugs the system currently does or doesn't have --
        a threshold no real run can ever satisfy must still fail."""
        import importlib
        import sys
        from pathlib import Path

        scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        evaluate = importlib.import_module("evaluate")

        impossible_gates = tmp_path / "impossible_gates.json"
        impossible_gates.write_text(json.dumps({
            "safety_pass_rate_min": 1.1, "unsupported_claim_rate_max": 0.0, "exhaustive_completeness_min": 1.0,
        }))
        out_path = tmp_path / "results.json"
        report_path = tmp_path / "report.md"
        exit_code = evaluate.main([
            "--gates", str(impossible_gates), "--out", str(out_path), "--report-out", str(report_path),
        ])
        assert exit_code != 0
