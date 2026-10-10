# Retriever alone

Do the retrieved policy chunks fit the question and the golden answer? Real `RetrievalEngine.retrieve()` output, scored against an independently-built golden answer.

Judge model: `gpt-4.1`  
14/16 cases exercised.

## DeepEval metrics (Contextual Precision / Recall / Relevancy)

| Metric | Cases | Exercised | Mean score | Pass rate |
|---|---:|---:|---:|---:|
| ContextualPrecisionMetric | 14 | 14 | 0.699 | 71.4% |
| ContextualRecallMetric | 14 | 14 | 0.714 | 85.7% |
| ContextualRelevancyMetric | 11 | 11 | 0.819 | 90.9% |

## Not exercised

- `nf_noise_level_io9` (missing_attribute): retriever returned nothing for this query
- `nf_competitor_not_in_corpus_quip` (out_of_corpus_brand): retriever returned nothing for this query
