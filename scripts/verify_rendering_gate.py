#!/usr/bin/env python3
"""Rendering gate for Philips specification / feature text.

Philips ships its "Technical Specifications" and feature panels as embedded
JSON that the page renders only when the section is opened. Evidence is
"visible page text only", so before JSON-derived spec text may be cited we
check, once, in a headless browser that opening the toggle really puts those
strings into the rendered (visible) text. The result is written to
``data/raw/rendering_gate.json`` and read by Step 2:

  philips_spec_expanded = true  -> spec/feature JSON text is ``expanded`` evidence
  philips_spec_expanded = false -> it is ``cross_check_only`` (kept, flagged, not cited)

Usage:
    python scripts/verify_rendering_gate.py [--url <philips pdp url>]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "rendering_gate.json"
DEFAULT_URL = "https://www.usa.philips.com/c-p/HX4031_71/3100-rechargeable-toothbrush"
NEEDLES = ["Operating time", "Battery type", "Battery indicator", "SmarTimer", "Optimal Plaque C2"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_context(
            locale="en-US", viewport={"width": 1400, "height": 1000},
            user_agent="Mozilla/5.0 (compatible; ProductsResearch/1.0; research-data-acquisition)",
        ).new_page()
        page.goto(args.url, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(2500)
        before = page.evaluate("document.body.innerText").lower()
        page.evaluate(
            """() => { const s = document.querySelector('[data-testid="technical-specifications"]');
                      const b = s && s.querySelector('[role="button"], button, [aria-expanded]'); if (b) b.click(); }"""
        )
        page.wait_for_timeout(1500)
        after = page.evaluate("document.body.innerText").lower()
        browser.close()

    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "needles": NEEDLES,
        "visible_before_toggle": {n: n.lower() in before for n in NEEDLES},
        "visible_after_toggle": {n: n.lower() in after for n in NEEDLES},
    }
    result["philips_spec_expanded"] = all(result["visible_after_toggle"].values()) and not any(
        result["visible_before_toggle"].values()
    )
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["philips_spec_expanded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
