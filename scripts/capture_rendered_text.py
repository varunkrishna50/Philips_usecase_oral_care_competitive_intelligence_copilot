#!/usr/bin/env python3
"""Save the browser-rendered visible text of each product page.

Philips renders price, rating and availability (and, once the toggle is
opened, the technical-specification table) client-side, so the static HTML
has them only as embedded JSON. "Evidence is visible page text" therefore
needs a rendered snapshot: for every Philips record this script loads the
page in a headless browser, opens the "Technical Specifications" toggle and
writes ``document.body.innerText`` to

    data/raw/text/philips/<product_id>.rendered.txt

Oral-B pages are server-rendered, but their stock status is a button:
``Add to cart ($329.99)`` is the *value* of an ``<input>`` that ``innerText``
omits, and an unavailable product shows ``Find A Retailer`` instead. For
every Oral-B record the script writes ``document.body.innerText`` plus a
labelled ``=== Buy box ===`` block holding the visible label(s) of the main
buy box:

    data/raw/text/oralb/<product_id>.rendered.txt

The path is recorded in the product JSON (``rendered_text_path``). Step 2 uses
it to cite price / rating / availability from visible text. Requests are
throttled per host (Philips 2 s, Oral-B 10 s).

Usage:
    python scripts/capture_rendered_text.py [--brand philips|oralb]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acquisition.base import portable_html_path  # noqa: E402

RAW = ROOT / "data" / "raw"
USER_AGENT = "Mozilla/5.0 (compatible; ProductsResearch/1.0; research-data-acquisition)"
DELAY = {"philips": 2, "oralb": 10}

OPEN_SPECS_JS = """() => { const s = document.querySelector('[data-testid="technical-specifications"]');
                          const b = s && s.querySelector('[role="button"], button, [aria-expanded]'); if (b) b.click(); }"""

# The main buy box: <div class="form-action"> holds either the add-to-cart <input>
# (label in its value) or the out-of-stock container ("Find A Retailer").
BUY_BOX_JS = """() => {
  const lines = [];
  const visible = (e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0
    && getComputedStyle(e).visibility !== 'hidden'; };
  const root = document.querySelector('.productView') || document.body;
  for (const box of root.querySelectorAll('div.form-action')) {
    if (!visible(box) || box.closest('.card, [class*=related], [class*=carousel], [class*=recommend]')) continue;
    const text = (box.innerText || '').trim();
    if (text) lines.push(text);
    for (const inp of box.querySelectorAll('input[type=submit], input[type=button]')) {
      const v = (inp.value || '').trim(); if (v && visible(inp)) lines.push(v);
    }
    if (lines.length) break;
  }
  return [...new Set(lines.flatMap(l => l.split('\\n').map(x => x.trim()).filter(Boolean)))];
}"""


def capture(browser, record: dict, brand: str) -> str:
    page = browser.new_context(
        locale="en-US", viewport={"width": 1400, "height": 1000}, user_agent=USER_AGENT,
    ).new_page()
    page.goto(record["product_url"], wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(2500)
    if brand == "philips":
        page.evaluate(OPEN_SPECS_JS)
        page.wait_for_timeout(1200)
    text = page.evaluate("document.body.innerText")
    if brand == "oralb":
        labels = page.evaluate(BUY_BOX_JS)
        text += "\n\n=== Buy box ===\n" + "\n".join(labels) + "\n"
    page.context.close()
    return text


def main() -> int:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--brand", choices=["philips", "oralb"], help="only this brand (default: both)")
    args = parser.parse_args()
    brands = [args.brand] if args.brand else ["philips", "oralb"]

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for brand in brands:
            for path in sorted((RAW / brand).glob(f"{brand}_*.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                text = capture(browser, record, brand)
                out = RAW / "text" / brand / f"{path.stem}.rendered.txt"
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("w", encoding="utf-8", newline="") as f:
                    f.write(text)
                record["rendered_text_path"] = portable_html_path(out, RAW / "html" / brand)
                path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"{path.stem}: {len(text)} chars")
                time.sleep(DELAY[brand])
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
