# Oral Care Competitive Intelligence Copilot

A retrieval-augmented question-answering system that compares Philips Sonicare
and Oral-B electric toothbrushes using only facts published on each brand's
own official US product pages. Every answer is grounded in a cited source;
the system refuses rather than guesses when a fact isn't available, a
question falls outside its scope, or a named product doesn't exist in its
catalog.

The project is a full pipeline, not just a chatbot: scraping and acquisition,
deterministic data normalization, hybrid (BM25 + semantic) retrieval, grounded
generation with a deterministic no-LLM fallback, and an automated evaluation
harness built on [DeepEval](https://github.com/confident-ai/deepeval).

For the detailed design rationale behind each stage — data formats, the
retrieval architecture, known limitations, and why specific choices were
made — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). This document
covers everything needed to set the project up and run it.

## Contents

- [Requirements](#requirements)
- [Environment setup](#environment-setup)
- [Project layout](#project-layout)
- [Running the pipeline end to end](#running-the-pipeline-end-to-end)
- [Running the application](#running-the-application)
- [Usage notes and known limitations](#usage-notes-and-known-limitations)
- [Running the tests](#running-the-tests)
- [Running the evaluation harness](#running-the-evaluation-harness)
- [Output locations at a glance](#output-locations-at-a-glance)
- [Troubleshooting](#troubleshooting)

## Requirements

- Python 3.11 or later
- macOS, Linux, or Windows (the data acquisition and index builds have been
  exercised primarily on macOS and Linux)
- No GPU required; embeddings run on CPU via `sentence-transformers`
- An OpenAI API key is **optional**. Without one, the application and the
  evaluation harness both run fully offline against a deterministic
  fallback — see [Usage notes](#usage-notes-and-known-limitations)

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

Playwright is included as a fallback for JavaScript-rendered pages. Neither
approved source currently requires it, but if you want the fallback usable,
install a browser once:

```bash
playwright install chromium
```

### Configuring an API key (optional)

```bash
cp .env.example .env
```

Open `.env` and set `OPENAI_API_KEY` to a real key if you want the
LLM-authored answer path exercised. `.env` is gitignored and never committed.
Leave it blank to run entirely on the deterministic fallback — nothing else
needs to change.

## Project layout

```
scripts/        Pipeline entry points (acquisition, normalization, indexing,
                 evaluation) — run these directly, in the order below
src/
  acquisition/   Step 1 — scraping and raw page capture
  normalization/ Step 2 — deterministic extraction into a comparison dataset
  rag/           Step 3 — retrieval, generation, citations, safety/scope
  evaluation/    DeepEval-based evaluation harness
  app/           Shared engine/client construction used by app.py
app.py           Streamlit UI
data/
  raw/           Step 1 output (scraped HTML/JSON snapshots, manifest)
  processed/     Step 2 output (comparison table, evidence, snapshots)
  index/         Step 3 output (FAISS index, retrieval corpus)
  evaluation/    Test cases, quality gates, generated evaluation reports
docs/            Architecture notes, the original case-study brief, and
                 review-feedback notes referenced by regression tests
tests/           Full automated test suite (pytest)
```

## Running the pipeline end to end

Each stage reads only the previous stage's output and writes to its own
`data/` subdirectory — nothing re-scrapes or re-derives data a later stage
doesn't need. Run them in order from a fresh checkout:

```bash
# 1. Acquire raw product pages (writes data/raw/)
python scripts/build_data.py

# 2. Normalize into a grounded comparison dataset (writes data/processed/)
python scripts/build_table.py

# 2a. Optional but recommended: validate Step 2's output before continuing
python scripts/validate_table.py

# 3. Build the retrieval corpus and the semantic index (writes data/index/)
python scripts/build_rag_corpus.py
python scripts/build_index.py
```

After this, `data/index/` holds everything the application needs to answer
questions — no network access is required again unless you re-run
acquisition.

## Running the application

Both entry points below call the same unmodified `answer_query()` pipeline —
answers, citations, and refusal behavior are identical either way.

### Command line

```bash
python -m src.rag.pipeline --query "What is the price of the Sonicare Prestige 9900?"
python -m src.rag.pipeline --query "Which products have 7 brushing modes?" --debug
python -m src.rag.pipeline --query "..." --no-llm   # force the deterministic fallback
```

### Streamlit UI

```bash
streamlit run app.py
```

The first question in a given server session takes a few extra seconds while
the embedding model and index load; every question after that reuses the
same cached engine. Each answer includes a **Sources** section linking to the
exact official product page(s) it was grounded in, and an optional debug
panel showing the detected intent, matched product, and citation validity
for each statement — never raw prompts or an API key.

## Usage notes and known limitations

- **The copilot only knows about Philips Sonicare and Oral-B products present
  in its catalog**, acquired from the two brands' official US product pages.
  It will not answer questions about other brands, other regions, or
  third-party retailer listings, and it will say so rather than improvise.
- **Give the exact product name when comparing two products.** Several
  models share most of their name with a close sibling (for example, *Oral-B
  iO Series 9* and *Oral-B iO Series 9 Genius* are two different products).
  To get a comparison resolved correctly and unambiguously, use the full,
  exact product name as it appears on the product page — a partial or
  approximate name is the single most common cause of an "I found more than
  one matching product" clarification response. This version of the system
  is built and tested against exact names; broader fuzzy matching for
  comparisons is not guaranteed.
- The system never gives medical, dental, or purchasing-recommendation
  advice, and never reveals or discusses its own instructions — these
  requests are refused by design, not by omission.

## Running the tests

```bash
pytest
```

The full suite (500+ tests) runs offline by default: a session-wide fixture
strips any configured `OPENAI_API_KEY` before the suite starts, so no test
ever makes a real, billed API call regardless of local `.env` configuration.
Expect the first run to take a minute or two while the embedding model loads
once for the session-scoped retrieval engine fixture.

To exercise the real OpenAI API manually (never run automatically, and never
part of this suite):

```bash
python scripts/verify_llm.py
```

## Running the evaluation harness

The evaluation harness (`src/evaluation/`) scores four independent layers —
retriever, generator, full pipeline, and final application output — using
DeepEval's built-in metrics and classifiers, plus a small number of custom
GEval criteria documented in `src/evaluation/`.

```bash
# Offline: deterministic checks only, no API key used, safe to run anywhere
python scripts/evaluate.py

# Live: real application LLM calls + a real DeepEval judge model (gpt-4.1
# by default) — this makes billed API calls
python scripts/evaluate.py --live

# Run just one category (repeatable)
python scripts/evaluate.py --live --category application
```

Each invocation prints a PASS/FAIL line per quality gate
(`data/evaluation/quality_gates.json`) and exits non-zero if any gate for a
category that was actually run fails. A category not requested is skipped,
not failed.

Two additional, narrower evaluation tools:

```bash
# Retrieval-only ablation: BM25-only vs. semantic-only vs. hybrid vs.
# hybrid+reranker, Recall@5/@10 — no LLM, no DeepEval judge
python scripts/evaluate_retrieval.py

# Save the current offline/live answer-quality numbers as the baseline to
# compare future changes against (run once, before changing behavior)
python scripts/save_answer_quality_baseline.py
```

## Output locations at a glance

| What | Where |
|---|---|
| Raw scraped pages, discovery manifest | `data/raw/` |
| Normalized comparison table, evidence, per-product snapshots | `data/processed/` |
| Retrieval corpus, FAISS index | `data/index/` |
| Test case definitions, quality gate thresholds | `data/evaluation/test_cases.json`, `data/evaluation/quality_gates.json` |
| Generated evaluation reports (offline) | `data/evaluation/deepeval/` — `report.json`, `report.md`, one `.md` per category |
| Generated evaluation reports (`--live`) | `data/evaluation/deepeval_live/` (same layout) |
| Saved answer-quality baseline | `data/evaluation/baseline_answer_quality.json` |

The two generated-report directories are gitignored: they're rewritten by
every evaluation run, so nothing meaningful is lost by not tracking them.

## Troubleshooting

- **`ModuleNotFoundError` when running a script directly** — run scripts from
  the repository root (`python scripts/build_data.py`, not from inside
  `scripts/`), and make sure the virtual environment is activated.
- **A question about a product that should exist comes back "not found"** —
  confirm the product name matches the catalog exactly; see
  [Usage notes](#usage-notes-and-known-limitations).
- **Evaluation `--live` fails immediately** — check that `OPENAI_API_KEY` is
  set in `.env` or your shell environment; offline mode (the default) never
  needs it.
