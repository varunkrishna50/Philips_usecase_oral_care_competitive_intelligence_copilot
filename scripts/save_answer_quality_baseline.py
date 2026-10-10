#!/usr/bin/env python3
"""Save the answer-quality numbers of the latest offline and live evaluation runs as the
'before' baseline (data/evaluation/baseline_answer_quality.json).

Run it ONCE, after an evaluation run of the behaviour you want to compare against and before
changing that behaviour. It refuses to overwrite an existing baseline unless --force is given,
so a later 'improved' run cannot silently replace the 'before' numbers.

    python scripts/evaluate.py --category application && python scripts/evaluate.py --live --category application
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
    for mode, dirname in (("offline", "deepeval"), ("live", "deepeval_live")):
        path = EVAL / dirname / "report.json"
        if not path.exists():
            print(f"missing {path}; run scripts/evaluate.py{' --live' if mode == 'live' else ''} --category application first")
            return 1
        results = json.loads(path.read_text(encoding="utf-8"))
        application = results.get("application")
        if not application:
            print(f"{path} has no 'application' section; re-run with --category application (or no --category)")
            return 1
        baseline[mode] = {
            **application["answer_quality_aggregate"],
            "template_samples": application["template_samples_checked"],
            "templates_with_jargon": len(application["template_jargon"]),
            "run_timestamp": results["timestamp"],
            "cases_total": application["total_cases"],
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
