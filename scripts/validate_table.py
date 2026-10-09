#!/usr/bin/env python3
"""Quality gate over the written comparison outputs (data/processed).

Checks the files themselves - the wide CSV, the long tables, evidence.json and
the saved text snapshots - so it validates exactly what a reviewer or a
downstream step will read. Errors exit non-zero; warnings are listed.

Usage:
    python scripts/validate_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from normalization.quality_gate import check_files  # noqa: E402


def main() -> int:
    errors, warnings = check_files(ROOT / "data" / "processed", ROOT / "data" / "raw", ROOT)
    lines = [f"ERROR   {e}" for e in errors] + [f"warning {w}" for w in warnings]
    lines.append(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    print("\n".join(lines))
    # keep the report next to the tables so it can be shared and reviewed
    (ROOT / "data" / "processed" / "validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
