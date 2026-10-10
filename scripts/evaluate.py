#!/usr/bin/env python3
"""DeepEval-based evaluation harness CLI: 4 independent eval categories
(retriever, generator, pipeline, application).

    python scripts/evaluate.py                       # offline, deterministic parts only, no API key used
    python scripts/evaluate.py --live                # real app LLM + real DeepEval judge (GPT-4.1 by default)
    python scripts/evaluate.py --live --category retriever    # just one category

Writes data/evaluation/deepeval/<category>.json + <category>.md per
category run, plus data/evaluation/deepeval/report.json (the full
combined dict) and report.md (an index linking every category run).
Prints PASS/FAIL per quality gate and exits non-zero if any gate for a
category that WAS run this invocation fails (a category not requested
is skipped, not failed -- see src/evaluation/quality_gates.py).

Never makes a real app or judge-model API call in offline mode
(default). --live uses whatever OPENAI_API_KEY/LLM_MODEL/
DEEPEVAL_JUDGE_MODEL are configured (via .env or the real environment)
-- never prints a key, never substitutes a different model.
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

from src.evaluation.report import render_category_report, render_index  # noqa: E402
from src.evaluation.runner import ALL_CATEGORIES, run_deepeval_suite  # noqa: E402

OUT_DIR = ROOT / "data" / "evaluation" / "deepeval"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true",
                         help="Use the real OpenAI API for both app generation and the DeepEval judge. "
                              "Default: offline/deterministic, no API key used.")
    parser.add_argument("--category", choices=ALL_CATEGORIES, action="append", dest="categories",
                         help="Run only this category (repeatable). Default: all 4.")
    parser.add_argument("--test-cases", default=str(ROOT / "data" / "evaluation" / "test_cases.json"))
    parser.add_argument("--gates", default=None, help="Path to a quality_gates.json override.")
    parser.add_argument("--out-dir", default=None, help=f"Default: {OUT_DIR} (offline) or {OUT_DIR}_live (--live).")
    args = parser.parse_args(argv)

    mode = "live" if args.live else "offline"
    out_dir = Path(args.out_dir) if args.out_dir else (OUT_DIR if mode == "offline" else OUT_DIR.with_name("deepeval_live"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running DeepEval evaluation in {mode!r} mode"
          + (f" (categories: {', '.join(args.categories)})" if args.categories else " (all 4 categories)") + "...")
    report = run_deepeval_suite(mode=mode, test_cases_path=args.test_cases, gates_path=args.gates, categories=args.categories)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "report.md").write_text(render_index(report), encoding="utf-8")
    for category in (args.categories or ALL_CATEGORIES):
        if category in report:
            (out_dir / f"{category}.md").write_text(render_category_report(report, category), encoding="utf-8")
    print(f"Wrote {out_dir}/report.{{json,md}} and one .md per category run")

    for gate in report["quality_gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        actual = f"{gate['actual']:.4f}" if isinstance(gate["actual"], float) else str(gate["actual"])
        print(f"GATE [{status}] {gate['name']}: actual={actual} threshold({gate['comparison']})={gate['threshold']}")

    if report["quality_gates_all_passed"]:
        print("\nPASS")
        return 0
    print("\nFAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
