#!/usr/bin/env python3
"""Step 3a: build the FAISS semantic index from the retrieval corpus.

Reads data/index/retrieval_documents.json (see build_rag_corpus.py),
writes data/index/faiss.index + data/index/metadata.json. Requires
network access on first run only, to download the pinned embedding
model (sentence-transformers/all-MiniLM-L6-v2) from the HuggingFace Hub;
it's cached locally afterwards.

Usage:
    python scripts/build_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, FAISS_METADATA_PATH, RETRIEVAL_DOCUMENTS_PATH  # noqa: E402
from src.rag.embedding_index import EmbeddingIndex  # noqa: E402
from src.rag.models import RetrievalDocument  # noqa: E402


def main() -> int:
    if not RETRIEVAL_DOCUMENTS_PATH.exists():
        print(f"{RETRIEVAL_DOCUMENTS_PATH} not found -- run scripts/build_rag_corpus.py first.")
        return 1

    raw = json.loads(RETRIEVAL_DOCUMENTS_PATH.read_text(encoding="utf-8"))
    documents = [RetrievalDocument.from_dict(d) for d in raw]

    print(f"Embedding {len(documents)} documents with {EMBEDDING_MODEL_NAME} ...")
    index = EmbeddingIndex()
    index.build(documents)
    index.save(FAISS_INDEX_PATH, FAISS_METADATA_PATH)

    print(f"FAISS index: {FAISS_INDEX_PATH} ({index.index.ntotal} vectors, dim={index.index.d})")
    print(f"Metadata: {FAISS_METADATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
