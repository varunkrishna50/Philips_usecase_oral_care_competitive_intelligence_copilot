# Step 1: Data Acquisition — Philips Sonicare vs. Oral-B Electric Toothbrushes

This is **Step 1 only** of the Philips vs. Oral-B competitive-intelligence
copilot: raw evidence acquisition from the two approved official US product
pages. It does **not** do normalization, embeddings, a vector DB, RAG, or
any LLM calls — that's later steps.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Playwright is included as a JS-rendering fallback (see "How dynamic pages
# are handled" below). It is not required by either site today, but if you
# want the fallback to actually be able to render a page, install a browser:
playwright install chromium
```

## How to run acquisition

```bash
python scripts/build_data.py
```

Options:

- `--refresh` — ignore the on-disk cache and re-fetch every listing and product page.
- `--target-per-brand N` — preferred number of distinct models to discover per brand before stopping discovery (default `10`; the pipeline still only needs 8 to meet the acceptance bar).
- `--verbose` — debug-level logging.

Console output looks like:

```
Philips:
  discovered: 11
  successfully scraped: 11
  failed: 0
  excluded (accessories/bundles/non-US/etc.): 0

Oral-B:
  discovered: 13
  successfully scraped: 13
  failed: 0
  excluded (accessories/bundles/non-US/etc.): 14
    - accessory_manual_or_kids_path_excluded: 9
    - category_or_collection_landing_page: 5

Total successfully acquired: 24
Full discovery report (discovered/excluded/scraped/failed URLs): data/raw/manifest.json
```

A full discovery/acquisition report — every discovered URL, every excluded
URL with its reason, every scraped URL, every failed URL, and per-brand
counts — is written to `data/raw/manifest.json`. A run log is appended to
`data/raw/acquisition.log`.

## Where raw data is stored

```
data/raw/
  philips/philips_<sku>.json       # one JSON record per Philips model
  oralb/oralb_<model-slug>.json    # one JSON record per Oral-B model
  html/philips/philips_<sku>.html  # raw HTML exactly as retrieved
  html/oralb/oralb_<model-slug>.html
  html/<brand>/_listing_<url-slug>.html  # cached category/listing pages
  manifest.json                    # discovery + acquisition report
  acquisition.log                  # run log
```

Filenames are derived deterministically from the product's SKU (Philips) or
normalized model slug (Oral-B) — never from discovery order — so re-running
the pipeline doesn't rename or reshuffle anything. Re-running without
`--refresh` reuses whatever is already on disk (see **Caching** below).

Each JSON record (`src/acquisition/models.py:ProductRecord`) contains, among
other things: `brand`, `product_name`, `product_url` (final canonical URL),
`discovered_url` (the URL exactly as found during discovery, pre-redirect),
`redirect_chain`, `source_listing_url`, `retrieved_at`, `http_status`,
`scrape_method`, `raw_html_path`, `content_hash` (SHA-256 of the saved raw
HTML), `raw_text` (full cleaned page text), `sections` (ordered
`{heading, text}` list for citing specific snippets later), `structured_data`
(JSON-LD blocks + `<meta>` tags, plus a raw `price_text` for Oral-B), and
`variants` (sibling color/size URLs of the same model — see below). Nothing
is inferred: a feature not mentioned on the page is simply absent from the
record, never filled in as `false`/`null`-with-a-guess.

## How Philips products are discovered

1. Fetch `https://www.usa.philips.com/c-m-pe/electric-toothbrushes/latest`.
2. Keep only `<a href>` links matching the official product-detail pattern
   `/c-p/<SKU>_<SUFFIX>/<slug>` on `www.usa.philips.com`. This also
   positively excludes accessory pages like `/c-p/HX9094_65/a3-all-in-one-4-pack-brush-heads`
   by slug keyword (`brush-head`, `replacement`, `charger`, etc.) as a
   backstop, though none appeared on the main listing in practice.
3. If fewer than `--target-per-brand` distinct models are found, the
   discovery queue also follows any other `/c-m-pe/electric-toothbrushes/.../latest`
   or `/all` sub-category links found on the page (e.g. "Philips One by
   Sonicare"), so the pipeline doesn't hard-code a fixed set of listing URLs.
4. A page is only accepted as a real product once scraped: it must yield a
   `schema.org/Product` JSON-LD block with a `name` and an `offers.price`.
   No price → not treated as a product page (this is how a stray accessory
   or discontinued page would be caught even if it slipped past step 2).
5. **Variants**: on the category page, Philips shows exactly one URL per
   model (one SKU). Other color SKUs of the same model are only discoverable
   from the color-swatch links embedded on that model's own product page —
   they share the identical URL *slug*, differing only in the SKU code
   (e.g. `HX9990_11`/`_12`/`_13` all use
   `.../sonicare-9900-prestige-power-toothbrush-with-senseiq`). Those
   sibling SKU URLs are recorded in the `variants` list on the one scraped
   record; they are not fetched or counted as separate products. Their
   color name is left `null` (not fetching each one to look it up, and
   not guessing it from the SKU code).

Result on a real run: 11 distinct Philips models discovered and scraped
from the single listing page, 0 excluded.

## How Oral-B products are discovered

1. Fetch `https://oralb.com/en-us/products/compare/electric-toothbrushes`
   (this 301-redirects to the real listing,
   `.../products/electric-toothbrushes/electric-toothbrushes/`).
2. Keep only links matching `/en-us/products/electric-toothbrushes/<slug>/`
   on `oralb.com`. The page's own pagination links (`?page=1..4`) are
   followed automatically, up to a safety cap, rather than hard-coding a
   page count.
3. Positive + negative filtering on the discovered slugs:
   - Reject known category/collection landing pages that share the same
     URL prefix but aren't a single product (`electric-toothbrushes`,
     `genius-series`, `pro-series`, `smart-series`,
     `twin-packs-and-bundles`).
   - Reject anything under `/replacement-brush-heads/`,
     `/more-products/floss/`, `/more-products/water-flosser/`,
     `/toothbrushes/manual-toothbrushes/`, `/toothbrushes/clic`, or
     `/kids/` (a separate product line not reachable from the approved
     electric-toothbrushes listing's own results).
   - **Explicit acquisition decision — twin-packs/bundles are excluded
     from the primary product dataset.** Any slug containing `twin-pack`
     or `bundle` is dropped. Unlike color variants, a twin-pack is not a
     distinct model — it is two handles (of a model that already appears
     on its own) sold together. Treating it as a 25th "product" would just
     double-count an existing model rather than add a new comparison unit,
     so the standalone handle/variant is the product unit for comparison,
     and twin-packs are left out of Step 1's dataset entirely (not scraped,
     not stored).
   - Same positive check as Philips at scrape time: the page must yield a
     product name and a price signal (either JSON-LD `offers.price`, or —
     since Oral-B pages often have no Product JSON-LD — a `.price`
     element / `aria-label="Price of the product : $X"` in the static HTML).
4. **Variants**: unlike Philips, Oral-B's listing shows every color as its
   own separate URL. So variant grouping has to happen at discovery time:
   URLs are grouped by a normalized slug (strip a leading `oral-b-` brand
   prefix, then strip only known trailing color words — `aqua-marine`,
   `black-onyx`, `matte-white`, etc.). This is deliberately conservative:
   it only strips literal color tokens, so `io-series-9-rechargeable-...`
   and `io-series-9-genius-...` (a materially different, distinctly-named
   product) are **not** merged just because they share a prefix. One URL
   per group is scraped (the first alphabetically, for determinism); its
   page is then checked for its own authoritative color-swatch links
   (`aria-label`/link text/`alt` text), which are merged into the final
   `variants` list together with the other same-group URLs found at
   discovery time. The other colors' pages are not separately fetched —
   they're the same model, and Step 1 is about acquiring evidence for the
   model, not re-acquiring the same specs/claims once per color.

Result on a real run: 13 distinct Oral-B models discovered and scraped
across 5 listing pages (the redirect target + 4 paginated pages), 14 URLs
excluded (9 accessory/manual/kids, 5 category landing pages).

## How dynamic pages are handled

Both approved sites were inspected directly (`robots.txt` + raw HTTP
fetches) before writing any scraping code:

- **Philips** product pages embed a complete `schema.org/Product` JSON-LD
  block (name, SKU, price, description) plus ~15KB of server-rendered
  visible text (specs, "what's in the box", FAQs, reviews).
- **Oral-B** product pages have no JSON-LD on every page, but all content
  — name, price, description, specs, FAQs — is in static server-rendered
  HTML.

So **neither site needs JavaScript rendering**, and the pipeline uses
`requests` + `BeautifulSoup`/`lxml` for everything. A Playwright fallback
(`src/acquisition/base.py:try_playwright_fetch`) is still implemented per
the spec: if a fetched page has no `<title>`, no heading, and under 500
characters of visible text (`utils.needs_js_fallback`), the pipeline
retries that one URL with headless Chromium before giving up on it. This
code path is not expected to fire against either site today, but is there
in case a page changes to a client-rendered template later. If Playwright
isn't installed, this degrades to a clearly logged failure for that one
URL — it never crashes the run.

Neither site presented a CAPTCHA or Cloudflare challenge page during
development. The pipeline actively checks every response for
bot-protection markers (`utils.looks_like_bot_protection` — "Attention
Required", "Just a moment", "captcha", a short body on a 403/429/503,
etc.) and raises a distinct, logged failure for that URL if one is
detected. **It never attempts to bypass a CAPTCHA or challenge** — if that
ever happens across a whole brand's listing, discovery for that brand
returns a clear, loud failure in the log and manifest rather than silently
producing partial or fabricated data.

### Politeness / robots.txt

- Philips' `robots.txt` doesn't disallow the toothbrush category or
  product paths used here.
- Oral-B's `robots.txt` names several AI/data crawlers (including
  `ClaudeBot`, `Claude-Web`, `anthropic-ai`) with a `Crawl-delay: 10`. The
  pipeline identifies itself with a descriptive User-Agent and enforces a
  10-second minimum delay between requests to `oralb.com`
  (`RateLimiter` in `src/acquisition/utils.py`), and a 2-second delay to
  Philips.

## Caching

Each product's cache key is simply "does
`data/raw/<brand>/<brand>_<slug>.json` already exist". If it does and
`--refresh` wasn't passed, that file is loaded from disk and the network is
never touched for that product. Listing/category pages are cached the same
way, keyed by a slug of their URL, under `data/raw/html/<brand>/_listing_*.html`.
Re-running `python scripts/build_data.py` with no flags after a successful
run completes in well under a second, with zero HTTP requests.

## Known limitations

- **Color names for Philips variants are left `null`.** The category page
  only ever shows one SKU per model, and the sibling-SKU swatch links on a
  product page don't carry a visible color name in the markup we checked.
  Rather than guess from the SKU code, the color label is left unavailable.
- **Oral-B variant grouping is a slug heuristic**, not a guarantee. It's
  cross-checked against each scraped page's own authoritative color-swatch
  links, but a model whose discovery-time slug doesn't fit the
  `<base>-<color-tokens>` shape could in principle be mis-grouped. None of
  the ~13 Oral-B models encountered during development triggered this.
- **Twin-packs/bundles are excluded entirely**, as documented above and per
  explicit instruction — they are not represented anywhere in the raw
  dataset, including as variants.
- **Oral-B Kids electric toothbrushes are out of scope.** They live under a
  separate `/en-us/products/kids/` path not reachable from the approved
  `electric-toothbrushes` listing's own product results, so they're
  excluded the same way accessories are. (Philips Sonicare for Kids *is*
  included, only because it happens to appear directly in the main
  `/c-m-pe/electric-toothbrushes/latest` grid — not a deliberate inclusion
  rule.)
- **Section extraction is heading-driven, not visually-driven.** A long
  run of page content with no intervening `<h1>`–`<h4>` (common on Oral-B
  product pages, which often jump straight from the `<h1>` to a review/FAQ
  widget) ends up as one large section rather than several finer-grained
  ones. The full raw text is still preserved either way.
- **No normalization of any kind.** Prices, specs, and claims are kept as
  the literal text found on the page (e.g. Oral-B's price is stored as the
  raw string `"$169.99"`, not a float). That's intentional — see Step 2.

## Repository layout

```
src/acquisition/
  models.py    # Pydantic schemas (ProductRecord, Section, VariantRef, ...)
  utils.py     # HTTP fetch+retry, rate limiting, bot-protection detection,
               # generic HTML text/section/JSON-LD/meta extraction
  base.py      # fetch_and_prepare (requests -> Playwright fallback),
               # on-disk cache (load_cached_record/save_record)
  philips.py   # Philips-specific discovery + scrape
  oralb.py     # Oral-B-specific discovery + scrape (variant grouping)
scripts/build_data.py   # pipeline entrypoint
tests/test_acquisition.py  # unit tests against local fixtures (no network)
data/raw/               # output: see "Where raw data is stored" above
```
