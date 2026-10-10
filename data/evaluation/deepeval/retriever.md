# Retriever alone

Do the retrieved policy chunks fit the question and the golden answer? Real `RetrievalEngine.retrieve()` output, scored against an independently-built golden answer.

Judge model: `gpt-4.1`  
11/16 cases exercised.

## DeepEval metrics (Contextual Precision / Recall / Relevancy)

| Metric | Cases | Exercised | Mean score | Pass rate |
|---|---:|---:|---:|---:|
| ContextualPrecisionMetric | 11 | 0 | n/a | n/a |
| ContextualRecallMetric | 11 | 0 | n/a | n/a |
| ContextualRelevancyMetric | 11 | 0 | n/a | n/a |

## Not exercised

- `compare_5700_vs_7100_price_modes` (comparison): retriever returned nothing for this query
- `unknown_product_sonicare9999` (out_of_corpus): no document-level gold scope for this category -- not a retrieval question
- `unknown_product_io_series_15` (out_of_corpus): no document-level gold scope for this category -- not a retrieval question
- `safety_medical_advice_injection` (medical_advice): no document-level gold scope for this category -- not a retrieval question
- `safety_toxic_injection` (prompt_injection_user): no document-level gold scope for this category -- not a retrieval question
