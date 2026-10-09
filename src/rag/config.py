"""Pinned configuration for the retrieval layer. Nothing here is chosen
dynamically at runtime -- every model name/weight/path is explicit so a
rebuild is reproducible.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"
COMPARISON_TABLE_CSV = PROCESSED_DIR / "comparison_table.csv"
COMPARISON_TABLE_PARQUET = PROCESSED_DIR / "comparison_table.parquet"
EVIDENCE_JSON = PROCESSED_DIR / "evidence.json"
PRODUCT_ALIASES_CSV = PROCESSED_DIR / "product_aliases.csv"

INDEX_DIR = ROOT / "data" / "index"
RETRIEVAL_DOCUMENTS_PATH = INDEX_DIR / "retrieval_documents.json"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
FAISS_METADATA_PATH = INDEX_DIR / "metadata.json"

# Pinned per the Step 3a spec; not chosen dynamically. 384-dim output.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Hybrid score = ALPHA * BM25_norm + (1 - ALPHA) * Semantic_norm.
# Not claimed optimal -- see scripts/evaluate_retrieval.py's ablation for
# an empirical BM25-only vs semantic-only vs hybrid comparison.
DEFAULT_ALPHA = 0.5

DEFAULT_TOP_K = 10

# Only these two hosts are approved Step 1 sources. Evidence from
# anywhere else must never be returned by evidence_filter.
APPROVED_SOURCE_DOMAINS = {"www.usa.philips.com", "oralb.com"}
