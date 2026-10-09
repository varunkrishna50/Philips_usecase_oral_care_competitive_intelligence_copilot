"""Pydantic data models for the acquisition stage.

These models capture raw, un-normalized evidence only. Nothing here infers or
derives feature values that are not literally present on the source page.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Section(BaseModel):
    """One heading + the text that followed it, in page order."""

    heading: str
    text: str


class VariantRef(BaseModel):
    """A sibling color/size variant of a product, recorded but not scraped
    as a separate product."""

    url: str
    label: Optional[str] = None
    sku: Optional[str] = None


class Mode(BaseModel):
    """One brushing mode as named (and, if stated, described) on the page."""

    name: str
    description: Optional[str] = None


class Feature(BaseModel):
    """One marketed device feature as listed on the page."""

    name: str
    description: Optional[str] = None


class Disclaimer(BaseModel):
    """A footnote/qualifier the page attaches to claims.

    ``marker`` is the symbol run the claim text carries ("*", "**", ...);
    it is empty for qualifiers that have no marker. ``source`` is
    "product" (specific to this product), "section" (a footnote tied to one
    page section, e.g. the warranty tab) or "site_wide" (identical on every
    product page, so not tied to a claim).
    """

    marker: str = ""
    text: str
    source: str = "product"
    section: Optional[str] = None


class MarkedStatement(BaseModel):
    """A page statement that carries a footnote marker (e.g. "...plaque*"),
    with a label saying where on the page it appears; disclaimers are linked
    to these by marker."""

    label: str
    text: str


class ProductDetails(BaseModel):
    """Structured commerce/spec fields read directly from the page's embedded
    data (never inferred). A field the page does not state is None / empty,
    and the reason is noted in ``notes``.
    """

    currency: Optional[str] = None
    original_price: Optional[float] = None
    sale_price: Optional[float] = None
    discount_amount: Optional[float] = None
    discount_pct: Optional[float] = None
    promotion_text: Optional[str] = None
    price_source: Optional[str] = None

    color_count: Optional[int] = None
    colors: list[str] = Field(default_factory=list)

    mode_count: Optional[int] = None
    modes: list[Mode] = Field(default_factory=list)
    intensities: list[str] = Field(default_factory=list)

    features: list[Feature] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)

    # "What's in the box" as listed on the page (dimensions/weight lines
    # split out into their own fields).
    box_items: list[str] = Field(default_factory=list)
    dimensions: Optional[str] = None
    weight: Optional[str] = None

    # "in_stock" | "out_of_stock" | other schema.org availability, as the page states it.
    availability: Optional[str] = None
    # The visible lead paragraph under the product title (Oral-B div.short-description).
    short_description: Optional[str] = None
    # A price present in page data but not displayed to shoppers (e.g. out of stock).
    price_displayed: Optional[bool] = None

    disclaimers: list[Disclaimer] = Field(default_factory=list)
    marked_statements: list[MarkedStatement] = Field(default_factory=list)
    # Bold lead-in phrases of the description bullets (short, shopper-visible
    # specification terms), Oral-B only.
    key_terms: list[str] = Field(default_factory=list)

    # Average rating and number of ratings only -- no review text.
    rating_avg: Optional[float] = None
    rating_count: Optional[int] = None
    rating_scale: float = 5.0

    notes: list[str] = Field(default_factory=list)


class DiscoveredProduct(BaseModel):
    """One candidate product URL found during the discovery phase, before
    any page content has been fetched."""

    brand: str
    product_name: Optional[str] = None
    product_url: str
    discovered_from: str


class ExcludedUrl(BaseModel):
    """A URL that was seen during discovery but filtered out, with the
    reason, for the discovery report."""

    brand: str
    url: str
    reason: str
    discovered_from: str


class ProductRecord(BaseModel):
    """Full raw acquisition record for one product, as required by the
    source-provenance section of the spec."""

    brand: str
    product_name: str

    # The final URL the page was actually served from (after redirects).
    product_url: str
    # The URL exactly as found during discovery, before following redirects.
    discovered_url: str
    # Every URL visited while following redirects, in order
    # (discovered_url == redirect_chain[0] when redirects occurred).
    redirect_chain: list[str] = Field(default_factory=list)

    source_listing_url: str
    retrieved_at: str
    http_status: int
    scrape_method: str  # "requests" | "requests_cache" | "playwright"

    raw_html_path: str
    content_hash: str

    raw_text: str
    sections: list[Section] = Field(default_factory=list)

    structured_data: dict[str, Any] = Field(default_factory=dict)
    variants: list[VariantRef] = Field(default_factory=list)
    # None only for records scraped before structured extraction existed;
    # load_cached_record treats those as stale and re-fetches them.
    details: Optional[ProductDetails] = None

    errors: list[str] = Field(default_factory=list)


class FailedUrl(BaseModel):
    brand: str
    url: str
    source_listing_url: str
    reason: str
    http_status: Optional[int] = None
