"""Unit tests for Step 1 (data acquisition).

All tests use local fixture HTML / fake HTTP sessions -- none of them hit
the live Philips or Oral-B sites. Fixtures live under tests/fixtures/.
"""
from __future__ import annotations

import pytest
import requests
from bs4 import BeautifulSoup
from pydantic import ValidationError

from acquisition import oralb, philips
from acquisition.models import ProductRecord, Section, VariantRef
from acquisition.utils import (
    BotProtectionDetected,
    RateLimiter,
    clean_soup,
    extract_json_ld,
    extract_sections,
    fetch_url,
)

from conftest import FIXTURES_DIR


def load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# URL validation / product-href pattern matching
# ---------------------------------------------------------------------------

class TestUrlValidation:
    def test_philips_product_href_matches_valid_product_url(self):
        m = philips.PRODUCT_HREF_RE.match("/c-p/HX9990_13/sonicare-9900-prestige-power-toothbrush-with-senseiq")
        assert m is not None
        assert m.groups() == ("HX9990", "13", "sonicare-9900-prestige-power-toothbrush-with-senseiq")

    def test_philips_product_href_rejects_category_page(self):
        assert philips.PRODUCT_HREF_RE.match("/c-m-pe/electric-toothbrushes/latest") is None

    def test_philips_product_href_rejects_support_subpage(self):
        # /support is a sub-path of a product page, not the product page itself.
        assert philips.PRODUCT_HREF_RE.match(
            "/c-p/HX9990_13/sonicare-9900-prestige-power-toothbrush-with-senseiq/support"
        ) is None

    def test_oralb_product_href_matches_valid_product_url(self):
        m = oralb.PRODUCT_HREF_RE.match(
            "/en-us/products/electric-toothbrushes/oral-b-io-series-9-rechargeable-electric-toothbrush-aqua-marine/"
        )
        assert m is not None
        assert m.group(1) == "oral-b-io-series-9-rechargeable-electric-toothbrush-aqua-marine"

    def test_oralb_product_href_rejects_replacement_brush_heads(self):
        assert oralb.PRODUCT_HREF_RE.match("/en-us/products/replacement-brush-heads/") is None

    def test_oralb_accessory_prefixes_catch_manual_and_flosser_paths(self):
        assert "/en-us/products/toothbrushes/manual-toothbrushes/".startswith(oralb.ACCESSORY_PATH_PREFIXES)
        assert "/en-us/products/more-products/water-flosser/x/".startswith(oralb.ACCESSORY_PATH_PREFIXES)


# ---------------------------------------------------------------------------
# Deduplication, including color-variant grouping
# ---------------------------------------------------------------------------

class TestVariantGrouping:
    def test_oralb_same_model_different_colors_group_together(self):
        base_a = oralb.normalize_base_slug("oral-b-io-series-9-rechargeable-electric-toothbrush-aqua-marine")
        base_b = oralb.normalize_base_slug("oral-b-io-series-9-rechargeable-electric-toothbrush-black-onyx")
        assert base_a == base_b

    def test_oralb_materially_different_product_not_merged(self):
        # "...-genius-..." is a different, materially distinct model, not
        # just a color variant of plain Series 9.
        base_plain = oralb.normalize_base_slug("oral-b-io-series-9-rechargeable-electric-toothbrush-aqua-marine")
        base_genius = oralb.normalize_base_slug("oral-b-io-series-9-genius-rechargeable-electric-toothbrush-sapphire-blue")
        assert base_plain != base_genius

    def test_oralb_inconsistent_brand_prefix_still_groups(self):
        # Observed in the wild: some color URLs for the same model omit the
        # leading "oral-b-" token.
        base_with_prefix = oralb.normalize_base_slug("oral-b-io-series-7-electric-toothbrush-white-alabaster")
        base_without_prefix = oralb.normalize_base_slug("io-series-7-electric-toothbrush-black")
        assert base_with_prefix == base_without_prefix

    def test_philips_variants_grouped_by_identical_slug_different_sku(self):
        soup = clean_soup(load_fixture("philips_product_page.html"))
        variants = philips._extract_variants(
            soup, "https://www.usa.philips.com/c-p/HX9990_13/sonicare-9900-prestige-power-toothbrush-with-senseiq"
        )
        skus = {v.sku for v in variants}
        assert skus == {"HX9990_11", "HX9990_12"}
        # The current product's own SKU must not appear in its own variant list.
        assert "HX9990_13" not in skus

    def test_oralb_authoritative_variant_links_carry_color_labels(self):
        soup = clean_soup(load_fixture("oralb_product_page.html"))
        variants = oralb._extract_variant_links(
            soup,
            "https://oralb.com/en-us/products/shop-all/oral-b-io-series-9-rechargeable-electric-toothbrush-aqua-marine/",
        )
        labels = {v.label for v in variants}
        assert "Black Onyx" in labels
        assert "Rose Quartz" in labels

    def test_oralb_discover_groups_color_variants_and_follows_pagination(self, tmp_path, monkeypatch):
        """Regression test: pagination links on the Oral-B listing page
        share the exact URL *path* of the excluded 'electric-toothbrushes'
        landing page (they differ only by a ?page=N query string). This
        must still be followed, not misfiled as an excluded landing page.
        """
        pages = {
            oralb.CATEGORY_URL: BeautifulSoup(load_fixture("oralb_listing_page.html"), "lxml"),
            "https://oralb.com/en-us/products/electric-toothbrushes/electric-toothbrushes/?page=1": BeautifulSoup(
                load_fixture("oralb_listing_page.html"), "lxml"
            ),
            "https://oralb.com/en-us/products/electric-toothbrushes/electric-toothbrushes/?page=2": BeautifulSoup(
                load_fixture("oralb_listing_page2.html"), "lxml"
            ),
        }

        def fake_fetch_listing_page(url, session, rate_limiter, html_dir, refresh):
            assert url in pages, f"unexpected listing URL requested: {url}"
            return pages[url]

        monkeypatch.setattr(oralb, "fetch_listing_page", fake_fetch_listing_page)

        discovered, excluded, meta = oralb.discover(
            session=None, rate_limiter=None, html_dir=tmp_path, refresh=False, target_count=100
        )

        discovered_slugs = {_[1] for _ in [(d, d.product_url.rstrip("/").rsplit("/", 1)[-1]) for d in discovered]}
        # io-series-9 (2 color URLs on page 1) must collapse to ONE product.
        assert sum(1 for s in discovered_slugs if "io-series-9-rechargeable" in s) == 1
        # io-series-6 (page 1) and io-series-7 (only reachable via the ?page=2 link) must both appear.
        assert any("io-series-6" in s for s in discovered_slugs)
        assert any("io-series-7" in s for s in discovered_slugs)

        excluded_reasons = {e.reason for e in excluded}
        assert "category_or_collection_landing_page" in excluded_reasons
        assert "twin_pack_or_bundle_excluded_by_acquisition_policy" in excluded_reasons
        assert "accessory_manual_or_kids_path_excluded" in excluded_reasons


# ---------------------------------------------------------------------------
# Product record schema / handling missing fields
# ---------------------------------------------------------------------------

class TestProductRecordSchema:
    def _minimal_kwargs(self) -> dict:
        return dict(
            brand="Philips",
            product_name="Test Toothbrush",
            product_url="https://www.usa.philips.com/c-p/HX0000_00/test",
            discovered_url="https://www.usa.philips.com/c-p/HX0000_00/test",
            source_listing_url="https://www.usa.philips.com/c-m-pe/electric-toothbrushes/latest",
            retrieved_at="2026-01-01T00:00:00+00:00",
            http_status=200,
            scrape_method="requests",
            raw_html_path="data/raw/html/philips/philips_hx0000-00.html",
            content_hash="a" * 64,
            raw_text="Some text",
        )

    def test_minimal_record_is_valid_and_defaults_are_empty(self):
        record = ProductRecord(**self._minimal_kwargs())
        assert record.sections == []
        assert record.structured_data == {}
        assert record.variants == []
        assert record.errors == []
        assert record.redirect_chain == []

    def test_missing_required_field_raises(self):
        kwargs = self._minimal_kwargs()
        del kwargs["content_hash"]
        with pytest.raises(ValidationError):
            ProductRecord(**kwargs)

    def test_record_round_trips_through_json(self):
        record = ProductRecord(
            **self._minimal_kwargs(),
            sections=[Section(heading="Specs", text="Battery: 3 weeks")],
            variants=[VariantRef(url="https://example.com/x", label="Black", sku="HX0000_01")],
        )
        reloaded = ProductRecord.model_validate_json(record.model_dump_json())
        assert reloaded == record


# ---------------------------------------------------------------------------
# Structured metadata parsing (JSON-LD), including malformed input
# ---------------------------------------------------------------------------

class TestStructuredMetadata:
    def test_extract_json_ld_parses_product_offer(self):
        soup = BeautifulSoup(load_fixture("philips_product_page.html"), "lxml")
        blocks = extract_json_ld(soup)
        products = [b for b in blocks if b.get("@type") == "Product"]
        assert len(products) == 1
        assert products[0]["offers"]["price"] == 379.96

    def test_extract_json_ld_skips_malformed_block_without_raising(self):
        soup = BeautifulSoup(load_fixture("philips_product_page_malformed_jsonld.html"), "lxml")
        blocks = extract_json_ld(soup)  # must not raise
        assert blocks == []

    def test_extract_json_ld_handles_multiple_blocks(self):
        soup = BeautifulSoup(load_fixture("oralb_product_page.html"), "lxml")
        blocks = extract_json_ld(soup)
        types = {b.get("@type") for b in blocks}
        assert types == {"BreadcrumbList", "Product"}


# ---------------------------------------------------------------------------
# Section extraction -- must key off heading/text tag semantics, not
# document position (no nth-child-style brittleness).
# ---------------------------------------------------------------------------

class TestSectionExtraction:
    def test_sections_group_text_under_preceding_heading(self):
        html = """
        <body>
          <h1>Title</h1>
          <p>Intro text.</p>
          <h2>Specs</h2>
          <p>Battery: 2 weeks</p>
          <li>Waterproof</li>
        </body>
        """
        soup = clean_soup(html)
        sections = extract_sections(soup)
        headings = [s["heading"] for s in sections]
        assert headings == ["Title", "Specs"]
        assert sections[0]["text"] == "Intro text."
        assert "Battery: 2 weeks" in sections[1]["text"]
        assert "Waterproof" in sections[1]["text"]

    def test_sections_are_equivalent_across_different_wrapper_markup(self):
        """Two pages with the same heading->text sequence but different
        <div> wrapping/nesting must produce the same sections -- proving
        extraction is driven by tag semantics, not nth-child position.
        """
        html_a = "<body><h2>Specs</h2><p>Battery: 2 weeks</p></body>"
        html_b = "<body><div class='card'><div><h2>Specs</h2></div><div class='inner'><p>Battery: 2 weeks</p></div></div></body>"
        sections_a = extract_sections(clean_soup(html_a))
        sections_b = extract_sections(clean_soup(html_b))
        assert sections_a == sections_b

    def test_sections_skip_removed_boilerplate(self):
        html = """
        <body>
          <nav><a href="/x">Site nav link</a></nav>
          <h1>Product</h1>
          <p>Real content.</p>
          <footer>Copyright footer text</footer>
        </body>
        """
        sections = extract_sections(clean_soup(html))
        all_text = " ".join(s["text"] for s in sections)
        assert "Real content." in all_text
        assert "Site nav link" not in all_text
        assert "Copyright footer text" not in all_text


# ---------------------------------------------------------------------------
# HTTP fetch: retries, failure handling, bot-protection detection
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code: int, text: str = "<html></html>", url: str = "https://example.com/"):
        self.status_code = status_code
        self.text = text
        self.url = url
        self.history: list = []


class _FakeSession:
    """Replays a scripted sequence of responses/exceptions, one per .get() call."""

    def __init__(self, script: list):
        self._script = list(script)
        self.calls = 0

    def get(self, url, timeout=None, headers=None, allow_redirects=True):
        self.calls += 1
        if not self._script:
            raise AssertionError("fake session ran out of scripted responses")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class TestFetchUrlFailureHandling:
    def test_retries_then_succeeds_after_connection_error(self):
        session = _FakeSession([
            requests.exceptions.ConnectionError("boom"),
            _FakeResponse(200, text="<html>ok</html>", url="https://example.com/ok"),
        ])
        rate_limiter = RateLimiter(delays={}, default_delay=0.0)
        result = fetch_url("https://example.com/", session, rate_limiter, max_retries=3, backoff_seconds=0.0)
        assert result.status == 200
        assert session.calls == 2

    def test_persistent_connection_error_raises_request_exception_not_crash(self):
        session = _FakeSession([
            requests.exceptions.ConnectionError("1"),
            requests.exceptions.ConnectionError("2"),
            requests.exceptions.ConnectionError("3"),
        ])
        rate_limiter = RateLimiter(delays={}, default_delay=0.0)
        with pytest.raises(requests.exceptions.ConnectionError):
            fetch_url("https://example.com/", session, rate_limiter, max_retries=3, backoff_seconds=0.0)

    def test_persistent_5xx_raises_http_error_not_assertion_error(self):
        """A page that keeps returning 503 (no exception ever raised by the
        transport) must still surface as a normal, catchable HTTP error --
        not an AssertionError that would crash the whole pipeline.
        """
        long_plain_error_body = (
            "<html><body>Internal server error, please retry later. " + ("Lorem ipsum dolor sit amet. " * 100) + "</body></html>"
        )
        session = _FakeSession([_FakeResponse(503, text=long_plain_error_body) for _ in range(3)])
        rate_limiter = RateLimiter(delays={}, default_delay=0.0)
        with pytest.raises(requests.exceptions.RequestException):
            fetch_url("https://example.com/", session, rate_limiter, max_retries=3, backoff_seconds=0.0)

    def test_bot_protection_page_raises_bot_protection_detected(self):
        session = _FakeSession([
            _FakeResponse(403, text="<html><title>Attention Required! | Cloudflare</title></html>"),
        ])
        rate_limiter = RateLimiter(delays={}, default_delay=0.0)
        with pytest.raises(BotProtectionDetected):
            fetch_url("https://example.com/", session, rate_limiter, max_retries=3, backoff_seconds=0.0)


class TestBrandScrapeFailureHandling:
    def test_philips_scrape_reports_failure_without_raising_on_bot_protection(self, tmp_path, monkeypatch):
        from acquisition.models import DiscoveredProduct

        def fake_fetch_and_prepare(url, session, rate_limiter):
            raise BotProtectionDetected("simulated CAPTCHA")

        monkeypatch.setattr(philips, "fetch_and_prepare", fake_fetch_and_prepare)

        item = DiscoveredProduct(
            brand="Philips",
            product_name="Test",
            product_url="https://www.usa.philips.com/c-p/HX0000_00/test",
            discovered_from="https://www.usa.philips.com/c-m-pe/electric-toothbrushes/latest",
        )
        record, failure = philips.scrape(item, None, None, tmp_path, tmp_path, refresh=True)
        assert record is None
        assert failure is not None
        assert "bot_protection" in failure.reason

    def test_philips_scrape_rejects_page_without_price_as_not_a_product(self, tmp_path, monkeypatch):
        from acquisition.base import FetchOutcome
        from acquisition.models import DiscoveredProduct

        no_price_html = "<html><head><title>No price</title></head><body><h1>Some Accessory</h1></body></html>"
        soup = clean_soup(no_price_html)

        def fake_fetch_and_prepare(url, session, rate_limiter):
            return FetchOutcome(
                html=no_price_html,
                soup=soup,
                raw_text="Some Accessory",
                sections=[],
                structured_data={"json_ld": [], "meta_tags": {}},
                final_url=url,
                redirect_chain=[],
                status=200,
                method="requests",
            )

        monkeypatch.setattr(philips, "fetch_and_prepare", fake_fetch_and_prepare)

        item = philips.DiscoveredProduct(
            brand="Philips",
            product_name="Test",
            product_url="https://www.usa.philips.com/c-p/HX0000_00/test",
            discovered_from="https://www.usa.philips.com/c-m-pe/electric-toothbrushes/latest",
        )
        record, failure = philips.scrape(item, None, None, tmp_path, tmp_path, refresh=True)
        assert record is None
        assert failure is not None
        assert failure.reason == "no_price_offer_found_not_treated_as_product_page"
