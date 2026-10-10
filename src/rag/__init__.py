"""Step 3a: retrieval foundation for the competitive-intelligence copilot.

Reads only Step 2's frozen outputs (data/processed/comparison_table.csv,
data/processed/evidence.json) -- never Step 1 raw HTML, never the live
websites, and (in 3a) never an LLM. See retrieve.py for the orchestration
entry point and docs/ARCHITECTURE.md for the design rationale.
"""
import os

# faiss and torch (pulled in by sentence-transformers) each bundle their
# own OpenMP runtime; loading both in one process can abort on macOS
# with "OMP: Error #15" unless this is set before either is imported.
# Harmless on platforms where the conflict doesn't occur.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
