#!/usr/bin/env python3
"""Step 3C.1: evaluation harness CLI.

    python scripts/evaluate.py               # offline, deterministic, no API key used
    python scripts/evaluate.py --live         # real LLM calls via src/rag/llm_provider.py

Writes data/evaluation/results.json and data/evaluation/evaluation_report.md.
Prints PASS/FAIL per quality gate and exits non-zero if any gate fails.

Never makes an OpenAI API call in offline mode (default). --live uses
whatever OPENAI_API_KEY/LLM_MODEL are configured (via .env or the real
environment) -- never prints the key, never substitutes a different
model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.quality_gates import QualityGates, evaluate_gates  # noqa: E402
from src.evaluation.report import render_report  # noqa: E402
from src.evaluation.runner import run_evaluation  # noqa: E402

RESULTS_PATH = ROOT / "data" / "evaluation" / "results.json"
REPORT_PATH = ROOT / "data" / "evaluation" / "evaluation_report.md"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use the real OpenAI API (src/rag/llm_provider.py config). Default: offline/deterministic, no API key used.")
    parser.add_argument("--test-cases", default=str(ROOT / "data" / "evaluation" / "test_cases.json"))
    parser.add_argument("--gates", default=None, help="Path to a quality_gates.json override.")
    parser.add_argument("--out", default=None, help="Default: results.json (offline) or results_live.json (--live) -- kept separate so one mode never overwrites the other (3C.1 section 11).")
    parser.add_argument("--report-out", default=None, help="Default: evaluation_report.md (offline) or evaluation_report_live.md (--live).")
    args = parser.parse_args(argv)

    mode = "live" if args.live else "offline"
    out_path = args.out or str(RESULTS_PATH if mode == "offline" else RESULTS_PATH.with_name("results_live.json"))
    report_out_path = args.report_out or str(REPORT_PATH if mode == "offline" else REPORT_PATH.with_name("evaluation_report_live.md"))
    print(f"Running 3C.1 evaluation in {mode!r} mode...")
    results = run_evaluation(mode=mode, test_cases_path=args.test_cases, gates_path=args.gates)

    Path(out_path).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")

    report_text = render_report(results)
    Path(report_out_path).write_text(report_text, encoding="utf-8")
    print(f"Wrote {report_out_path}")

    agg = results["aggregate_metrics"]
    print(f"\n{agg['passed_cases']}/{agg['exercised_cases']} exercised cases passed "
          f"({agg['not_exercised_cases']} not_exercised).")

    for gate in results["quality_gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"GATE [{status}] {gate['name']}: actual={gate['actual']:.4f} threshold({gate['comparison']})={gate['threshold']}")

    if results["quality_gates_all_passed"]:
        print("\nPASS")
        return 0
    print("\nFAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
