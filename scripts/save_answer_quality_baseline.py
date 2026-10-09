#!/usr/bin/env python3
"""Save the answer-quality numbers of the latest offline and live evaluation runs as the
'before' baseline (data/evaluation/baseline_answer_quality.json).

Run it ONCE, after an evaluation run of the behaviour you want to compare against and before
changing that behaviour. It refuses to overwrite an existing baseline unless --force is given,
so a later 'improved' run cannot silently replace the 'before' numbers.

    python scripts/evaluate.py && python scripts/evaluate.py --live
    python scripts/save_answer_quality_baseline.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "evaluation"
OUT = EVAL / "baseline_answer_quality.json"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="overwrite an existing baseline")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        print(f"{OUT.name} already exists; it is the 'before' reference. Use --force to replace it.")
        return 1

    baseline: dict = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git("rev-parse", "--short", "HEAD"),
        "note": "Answer-quality numbers measured on the answers assembled BEFORE the de-duplication / plain-wording fixes.",
    }
    for mode, name in (("offline", "results.json"), ("live", "results_live.json")):
        path = EVAL / name
        if not path.exists():
            print(f"missing {name}; run scripts/evaluate.py{' --live' if mode == 'live' else ''} first")
            return 1
        results = json.loads(path.read_text(encoding="utf-8"))
        quality = results.get("answer_quality")
        if not quality:
            print(f"{name} has no answer_quality section; re-run the evaluation with the metrics in place")
            return 1
        baseline[mode] = {
            **quality["aggregate"],
            "template_samples": quality["template_samples"],
            "templates_with_jargon": len(quality["template_jargon"]),
            "run_timestamp": results["timestamp"],
            "cases_total": results["aggregate_metrics"]["total_cases"],
        }
    OUT.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for mode in ("offline", "live"):
        b = baseline[mode]
        print(f"  {mode}: duplicate sources {b['duplicate_sources']}/{b['sources_shown']}, "
              f"redundant segments {b['redundant_segments']}/{b['segments_shown']}, "
              f"answers with jargon {b['cases_with_jargon']}/{b['cases']}, templates with jargon {b['templates_with_jargon']}/{b['template_samples']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
