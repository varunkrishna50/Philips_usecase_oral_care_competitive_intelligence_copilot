# Full RAG pipeline

The full question flow -- contextualization, retrieval, reranking, generation -- scored for the RAG triad (Faithfulness, Answer Relevancy, Contextual Relevancy) plus a GEval Groundedness check against independently-loaded gold context. Must-abstain cases (out_of_corpus, medical_advice, prompt-injection) are included.

Judge model: `gpt-4.1`  
16/16 cases exercised.

## DeepEval metrics

| Metric | Cases | Exercised | Mean score | Pass rate |
|---|---:|---:|---:|---:|
| AnswerRelevancyMetric | 16 | 16 | 0.736 | 87.5% |
| FaithfulnessMetric | 16 | 8 | 1.000 | 100.0% |
| ContextualRelevancyMetric | 16 | 11 | 0.697 | 81.8% |
| Groundedness | 16 | 11 | 0.697 | 72.7% |

