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

    errors: list[str] = Field(default_factory=list)


class FailedUrl(BaseModel):
    brand: str
    url: str
    source_listing_url: str
    reason: str
    http_status: Optional[int] = None
