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
- An OpenAI API key is optional. Without one, the application and the
  evaluation harness run fully offline against a deterministic fallback —
  see [Usage notes](#usage-notes-and-known-limitations)

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

### Configuring an API key

```bash
cp .env.example .env
```

Open `.env` and set `OPENAI_API_KEY` to a real key
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
app.py           Streamlit chat UI
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
```

### Streamlit UI

```bash
streamlit run app.py
```

It's a chat interface: questions and answers persist in the session as a
conversation, with a Sources panel alongside that shows the cited product
page(s), section, and quote for whichever answer you last clicked a
citation marker on. A Debug/Retrieval details expander below the chat
shows the detected intent, matched product, and citation validity for the
most recent answer — never raw prompts or an API key.

The first question in a given server session takes a few extra seconds
while the embedding model and index load; every question after that reuses
the same cached engine.

## Usage notes and known limitations

- The copilot only knows about Philips Sonicare and Oral-B products in its
  catalog, acquired from the two brands' official US product pages. It
  won't answer questions about other brands, other regions, or retailer
  listings — it says so rather than improvise.
- Give the exact product name when comparing two products. Several models
  share most of their name with a close sibling (*Oral-B iO Series 9* vs.
  *Oral-B iO Series 9 Genius*, for example). A partial or approximate name
  is the most common cause of an "I found more than one matching product"
  clarification. Fuzzy matching across sibling variants isn't guaranteed.
- Attribute and intent detection (which attribute a question is about,
  whether it's a comparison or an exhaustive "which products" question)
  uses the LLM when a key is configured, with a deterministic regex
  fallback otherwise — same no-key-required guarantee as generation.
- Medical, dental, and purchasing-recommendation advice is refused by
  design, as is any request to reveal or discuss the system's own
  instructions.

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
python scripts/evaluate.py --live --category application
```

Each invocation prints a PASS/FAIL line per quality gate
(`data/evaluation/quality_gates.json`) and exits non-zero if any gate for a
category that was actually run fails. A category not requested is skipped,
not failed.

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

- `ModuleNotFoundError` when running a script directly — run scripts from
  the repository root (`python scripts/build_data.py`, not from inside
  `scripts/`), and make sure the virtual environment is activated.
- A question about a product that should exist comes back "not found" —
  confirm the product name matches the catalog exactly; see
  [Usage notes](#usage-notes-and-known-limitations).
- `evaluate.py --live` fails immediately — check that `OPENAI_API_KEY` is
  set in `.env` or the shell environment; offline mode (the default)
  doesn't need it.
