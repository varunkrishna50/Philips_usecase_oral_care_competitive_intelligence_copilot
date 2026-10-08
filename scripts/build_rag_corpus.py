#!/usr/bin/env python3
"""Step 3a: build the retrieval corpus from Step 2's frozen output.

Reads data/processed/comparison_table.parquet + evidence.json, writes
data/index/retrieval_documents.json. Does not touch embeddings/FAISS
(see scripts/build_index.py) and never re-scrapes anything.

Usage:
    python scripts/build_rag_corpus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.config import RETRIEVAL_DOCUMENTS_PATH  # noqa: E402
from src.rag.corpus import build_retrieval_documents, load_product_corpus  # noqa: E402


def main() -> int:
    product_corpus = load_product_corpus()
    documents = build_retrieval_documents(product_corpus=product_corpus)

    RETRIEVAL_DOCUMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_DOCUMENTS_PATH.write_text(
        json.dumps([d.to_dict() for d in documents], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    brands = sorted({p.brand for p in product_corpus})
    print(f"Products: {len(product_corpus)}")
    for brand in brands:
        print(f"  {brand}: {sum(1 for p in product_corpus if p.brand == brand)}")
    print(f"Retrieval documents: {len(documents)}")
    print(f"Written to: {RETRIEVAL_DOCUMENTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
