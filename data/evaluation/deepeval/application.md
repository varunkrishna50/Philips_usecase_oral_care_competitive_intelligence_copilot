# Application answer quality

Is the final, user-facing answer correct, complete, and safe? GEval Correctness/Completeness, Citation Faithfulness, comparison-arithmetic ground truth, a safety bundle (toxicity, PII/data leakage, prompt-injection resistance, role adherence, non-advice), plus deterministic content checks and readability.

Judge model: `gpt-4.1`  
16/16 cases exercised.

## Correctness / Completeness / Citation

| Metric | Cases | Exercised | Mean score | Pass rate |
|---|---:|---:|---:|---:|
| Correctness | 16 | 0 | n/a | n/a |
| Completeness | 16 | 0 | n/a | n/a |
| CitationFaithfulness | 16 | 0 | n/a | n/a |
| Toxicity | 16 | 0 | n/a | n/a |
| PIILeakage | 16 | 0 | n/a | n/a |
| DataLeakage | 16 | 0 | n/a | n/a |
| PromptInjection | 16 | 0 | n/a | n/a |
| RoleAdherence | 16 | 0 | n/a | n/a |
| NonAdvice | 16 | 0 | n/a | n/a |
| ComparisonArithmetic | 4 | 4 | 1.000 | 100.0% |

## Content checks (must_include / must_not_include)

5 case(s) declared phrases; 0 failed.

## Readability (answer_quality.py)

Duplicate-source rate: 0.0%; redundant-segment rate: 0.0%; jargon rate: 0.0% (0/11 cases). Templates with jargon: 0/16.

## Operational (latency / tokens)

0 case(s) made a real app-generation call (no latency recorded).


