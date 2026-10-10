# Full RAG pipeline

The full question flow -- contextualization, retrieval, reranking, generation -- scored for the RAG triad (Faithfulness, Answer Relevancy, Contextual Relevancy) plus a GEval Groundedness check against independently-loaded gold context. Must-abstain cases (out_of_corpus, medical_advice, prompt-injection) are included.

Judge model: `gpt-4.1`  
16/16 cases exercised.

## DeepEval metrics

| Metric | Cases | Exercised | Mean score | Pass rate |
|---|---:|---:|---:|---:|
| AnswerRelevancyMetric | 16 | 0 | n/a | n/a |
| FaithfulnessMetric | 16 | 0 | n/a | n/a |
| ContextualRelevancyMetric | 16 | 0 | n/a | n/a |
| Groundedness | 16 | 0 | n/a | n/a |

