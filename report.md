# Project Report

## Coverage against the case study brief

Case Study 1 (Oral Care Competitive Intelligence Copilot), scored against its
own stated rubric:

| Category | Weight | Delivered |
|---|---|---|
| GenAI quality & grounding | 40 | Evidence-constrained generation with per-segment citation validation, a deterministic no-LLM fallback with no loss of correctness, four-status phrasing (stated / stated-absent / not-mentioned / ambiguous) instead of guessing, prompt-injection and brand-preference refusals. |
| Data extraction & normalization | 25 | 21 products (10 Philips Sonicare, 11 Oral-B) scraped from the two approved US sources; deterministic field extraction into `comparison_table.csv` + `evidence.json`, every value traceable to its source sentence; a validation pass (`scripts/validate_table.py`) over the written output. |
| Engineering quality | 20 | Five independently runnable, independently testable stages; 500+ pytest tests; no test ever makes a real API call; typed dataclasses throughout; every LLM failure mode (no key, network error, bad JSON) degrades to the deterministic path rather than crashing. |
| Evaluation | 15 | DeepEval harness across four independent layers (retriever / generator / pipeline / application) with pass/fail quality gates, an offline mode and a `--live` mode, 16+ test cases exceeding the 15-question minimum. |
| **Bonus (+10)** | — | **(c) Advanced eval** — LLM-as-judge (`GEval` Correctness/Completeness/Groundedness/RoleAdherence) plus a dedicated citation-faithfulness metric, prompt-injection and non-advice classifiers. Also partially covers **(d)** — real per-call latency/token instrumentation and cost notes, see [Cost & latency](#cost--latency) below. MCP (a) and fine-tuning (b) were deliberately not pursued; see [Design choices](#design-choices). |

## Approach summary

The goal was a question-answering copilot that compares Philips Sonicare and
Oral-B electric toothbrushes using only facts published on each brand's own
official US product pages, with every answer traceable back to a specific
source and no invented or inferred facts.

The system is built as four sequential, independently runnable stages that
mirror the data's lifecycle, plus a parallel automated evaluation harness
that scores the result at every layer:

1. **Scrape** — discover and fetch product pages from the two approved
   sources only; save raw HTML/JSON snapshots and a full discovery manifest
   (what was found, scraped, excluded, and why).
2. **Store / normalize** — deterministically extract each page's facts
   (price, modes, battery life, included items, and so on) into a grounded,
   auditable comparison table with full evidence provenance. No LLM is
   involved in this step; every value can be traced back to the exact
   sentence or spec field it came from.
3. **Retrieval / tools** — build a hybrid BM25 + semantic (FAISS) index over
   the normalized evidence, and a query-understanding layer that resolves
   which product(s), attribute(s), and intent (direct question, comparison,
   or "which products have X") a question is asking about. Product
   matching is regex/tokenization over the closed catalog; attribute and
   intent classification are LLM-first with a regex fallback when no key
   is configured. A deterministic comparison engine handles
   attribute-by-attribute arithmetic.
4. **LLM (generation)** — a grounded-generation layer that either asks an
   LLM to phrase an answer strictly from the retrieved evidence (every
   statement must cite the evidence it came from) or, with no API key
   configured, falls back to a fully deterministic, evidence-grounded
   answer with no loss of correctness.
5. **Evaluation** — a DeepEval-based harness that scores the retriever, the
   generator, the full pipeline, and the final application output as four
   independent layers, so a retrieval bug and a generation bug are never
   confounded with each other.

Each stage reads only the previous stage's frozen output and never reaches
back to re-derive something a later stage already owns — Step 3 never
re-scrapes, Step 4 never re-normalizes, and so on. This keeps every stage
independently testable and lets a bug be localized to exactly one stage.

## Architecture / data flow

```
 +----------+     +----------+     +-------------------+     +----------+     +------------+
 |  SCRAPE  | --> |  STORE   | --> |  RETRIEVAL/TOOLS   | --> |   LLM    | --> | EVALUATION |
 +----------+     +----------+     +-------------------+     +----------+     +------------+
```

| Stage | Code | Data written | What happens |
|---|---|---|---|
| Scrape | `src/acquisition/`, `scripts/build_data.py` | `data/raw/` | Fetch Philips Sonicare US + Oral-B US product pages only; save HTML/JSON + a manifest of discovered/scraped/excluded URLs. |
| Store | `src/normalization/`, `scripts/build_table.py` | `data/processed/` | Deterministic extraction into a comparison table + `evidence.json` (every value traceable to its source sentence). No LLM. |
| Retrieval / tools | `src/rag/` (`retrieve.py`, `query_understanding.py`, `comparison_engine.py`, `scope.py`), `scripts/build_rag_corpus.py`, `scripts/build_index.py` | `data/index/` | Hybrid BM25 + semantic retrieval with reranking; LLM-first attribute/intent classification (regex fallback) plus regex product matching; safety & scope gates checked before retrieval; deterministic comparison arithmetic. |
| LLM | `src/rag/generation.py`, `src/rag/llm_provider.py` | — (served live) | Evidence-constrained prose generation with a deterministic no-LLM fallback; every LLM-authored statement's citations are validated against the evidence actually supplied before being shown. |
| Evaluation | `src/evaluation/`, `scripts/evaluate.py` | `data/evaluation/deepeval*/` | DeepEval built-in metrics/classifiers plus custom GEval criteria, scored separately for the retriever, the generator, the full pipeline, and final application output; an offline mode that never calls a real API, and an opt-in `--live` mode. |

Two user-facing entry points (`app.py` the Streamlit UI, and
`python -m src.rag.pipeline` the CLI) sit on top of stage 4 and call the
identical `answer_query()` function — there is no second, divergent
answering path.

## Design choices

- **Evidence-first, LLM-second.** Facts are extracted and verified
  deterministically before any LLM is involved (Store stage). The LLM only
  phrases or summarizes facts it's explicitly handed, never recalls or
  infers one on its own, and every statement it produces is checked
  against the evidence it cited before being shown.
- **Deterministic arithmetic for comparisons.** Numeric and boolean
  attribute comparisons (which product is cheaper, by how much, which has
  a feature) are computed in code — a language model judge can't verify
  arithmetic more reliably than the ground truth already does. The
  comparison table renders from these computed values; the LLM only writes
  a short, non-numeric takeaway on top of it.
- **LLM-first with a deterministic fallback everywhere.** Every answer type
  (direct question, exhaustive "which products have X", comparison) and
  the attribute/intent classification ahead of retrieval follow the same
  pattern: use the LLM when a key is configured and it succeeds, fall back
  to a fully grounded deterministic path otherwise. No API key never means
  "no answer."
- **Citation validation as a hard gate.** Every LLM-authored segment's
  citation IDs are checked against the evidence actually supplied; a
  segment whose citations don't resolve is dropped rather than shown with
  a fabricated source.
- **Safety and scope checked before product matching.** Prompt-probe and
  subjective "which brand is better" requests are refused ahead of
  retrieval, so a bare brand name in an off-purpose question is never
  mistaken for an ambiguous product question.
- **Status-aware phrasing.** The system distinguishes "the page doesn't
  mention this," "the page states it does not have this," "the page gives
  conflicting values," and "the page states this," and says which one
  applies rather than guessing.
- **Evaluation built on DeepEval**, using built-in metrics and classifiers
  wherever one fits, with a custom GEval criterion only where none exists
  (comparison arithmetic is still checked deterministically, not by an LLM
  judge). Golden answers and expected claims are hand-authored directly in
  `data/evaluation/test_cases.json` rather than mechanically derived from
  a separate gold corpus — a deliberate choice to maintain one source of
  truth for test cases instead of two that can drift apart.
- **No orchestration framework, no fine-tuning.** Retrieval, query
  understanding, and citation validation are plain Python modules wired
  together directly, not a LangChain/LangGraph pipeline — the problem is a
  four-stage linear flow with no branching agent behavior, so a framework
  would add indirection without adding capability. Similarly, no
  fine-tuning/adapters: a pinned low-temperature prompt with strict
  citation validation already gives consistent output style and is
  cheaper to iterate on than a training run. Both were available bonus
  options and were passed over by design, in favor of the advanced-eval
  and cost/latency bonus work described below.

## Cost & latency

Every LLM call (`src/rag/llm_provider.py`, `OpenAIStructuredClient`) records
its own latency and token usage (`last_latency_seconds`, `last_usage`).
During a `--live` run of the application evaluation category, these are
captured per test case and aggregated into an "Operational" section of the
generated report (`src/evaluation/application_eval.py`, rendered by
`src/evaluation/report.py`). A representative run
(`data/evaluation/deepeval_live/application.md`):

- 9 of 16 test cases triggered a real generation call (the rest were
  resolved deterministically — refusals, clarifications, comparisons).
- Mean latency: **2.27s** per generation call.
- Token usage: **17,095 total** (16,169 prompt / 926 completion) across
  those 9 calls — roughly 1,800 input tokens and 100 output tokens per
  call. The evidence payload handed to the model, not the answer itself,
  is the dominant cost driver.
- At public `gpt-4o-mini` pricing (~$0.15 / 1M input, $0.60 / 1M output
  tokens), that's on the order of **$0.0003 per answered query** — token
  counts are logged, not converted to a dollar figure in code, since no
  pricing table is maintained in the repo and prices change independently
  of it.

**What this doesn't yet cover:** the separate, smaller LLM call used for
query-understanding (`classify_query`, attribute/intent detection) resolves
its own client independently inside `retrieve()` rather than reusing the
generation client the evaluator meters, so its latency/tokens aren't
included in the aggregate above. The true per-query cost and latency are
therefore slightly higher than what's reported.

**Optimization notes:**
1. Prompt size, not output size, dominates cost (≈18x more tokens in than
   out) — trimming the evidence payload per call (shorter quotes, fewer
   retrieved passages) would cut cost more than a smaller model would.
2. No response caching exists — an identical or near-identical question
   re-triggers a full LLM round trip every time. A cache keyed on
   normalized query text would remove most of the latency and cost for
   repeat questions, with low staleness risk since the catalog only
   changes when acquisition is re-run.
3. `gpt-4o-mini` is already the smallest current-generation OpenAI model
   suitable for structured JSON output — model choice isn't the lever
   left to pull; prompt size and caching are.

## Limitations

- **Scope is intentionally narrow**: Philips Sonicare and Oral-B US official
  product pages only (~21 products at the time of writing). No retailer
  listings, no other regions, no other product categories.
- **Comparisons require the exact product name.** Several models share most
  of their name with a close sibling (*Oral-B iO Series 9* vs. *Oral-B iO
  Series 9 Genius*); the matcher resolves a name to the simplest product
  that needs no qualifier the question didn't mention, not a full
  fuzzy-matching system across sibling variants. A comparison naming one
  product not in the catalog is refused with an explicit message rather
  than silently answered about only the other product. See the README's
  usage notes.
- **No automatic freshness guarantee.** Prices, promotions, and stock status
  are snapshotted at scrape time; nothing re-fetches automatically between
  runs (`scripts/check_freshness.py` detects drift on demand but does not
  run on a schedule).
- **A markdown-table comparison answer is harder for a prose-oriented
  citation judge to verify** than a plain-sentence answer, even when the
  table's values are correct by construction (computed in code, not by the
  LLM) — a known gap between how the harness measures and how the
  production answer is actually formatted.
- **Single-user design.** Built for local use or a single Streamlit
  deployment, not concurrent high-throughput traffic or multi-tenant
  access control.
- **Medical/dental advice and brand-preference questions are refused by
  design**, not omission — a deliberate boundary, even though it's a
  class of question a general-purpose chatbot would normally answer.

For the detailed design rationale behind each stage — data formats, the
retrieval architecture in depth, and the reasoning behind specific
extraction and normalization rules — see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
