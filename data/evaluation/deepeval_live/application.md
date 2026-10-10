# Application answer quality

Is the final, user-facing answer correct, complete, and safe? GEval Correctness/Completeness, Citation Faithfulness, comparison-arithmetic ground truth, a safety bundle (toxicity, PII/data leakage, prompt-injection resistance, role adherence, non-advice), plus deterministic content checks and readability.

Judge model: `gpt-4.1`  
16/16 cases exercised.

## Correctness / Completeness / Citation

| Metric | Cases | Exercised | Mean score | Pass rate |
|---|---:|---:|---:|---:|
| Correctness | 16 | 16 | 0.657 | 68.8% |
| Completeness | 16 | 16 | 0.595 | 68.8% |
| CitationFaithfulness | 16 | 11 | 0.455 | 45.5% |
| Toxicity | 16 | 16 | 1.000 | 100.0% |
| PIILeakage | 16 | 16 | 1.000 | 100.0% |
| DataLeakage | 16 | 16 | n/a | 100.0% |
| PromptInjection | 16 | 16 | n/a | 100.0% |
| RoleAdherence | 16 | 16 | 0.856 | 93.8% |
| NonAdvice | 16 | 16 | 1.000 | 100.0% |
| ComparisonArithmetic | 3 | 3 | 1.000 | 100.0% |

## Content checks (must_include / must_not_include)

16 case(s) declared phrases; 11 failed.

## Readability (answer_quality.py)

Duplicate-source rate: 0.0%; redundant-segment rate: 0.0%; jargon rate: 18.2% (2/11 cases). Templates with jargon: 0/16.

## Operational (latency / tokens)

10 case(s) made a real app-generation call. Mean latency: 1.26s

Total tokens: 14495 (prompt: 13639, completion: 856). token counts, not a dollar cost -- no per-model pricing table exists in this repo.

