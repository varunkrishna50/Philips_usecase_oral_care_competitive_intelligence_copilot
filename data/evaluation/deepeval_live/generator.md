# Generator alone

Given known-good (golden) context -- retrieval bypassed entirely -- does the real generation hook produce a faithful, relevant, correctly-cited answer?

Judge model: `gpt-4.1`  
Mode: `live`  
10/16 cases exercised.

## DeepEval metrics (Faithfulness, Answer Relevancy, Citation Faithfulness)

| Metric | Cases | Exercised | Mean score | Pass rate |
|---|---:|---:|---:|---:|
| FaithfulnessMetric | 10 | 10 | 0.950 | 100.0% |
| AnswerRelevancyMetric | 10 | 10 | 1.000 | 100.0% |
| CitationFaithfulness | 10 | 10 | 0.700 | 70.0% |

## Not exercised

- `calc_price_diff_dc9000_vs_io9` (multi_hop_calculation): generator produced no usable segments on golden context
- `injection_unsupported_superiority_claim` (prompt_injection_unsupported_claim): no golden context for this category
- `safety_medical_bleeding_gums_cure` (medical_advice): no golden context for this category
- `nf_noise_level_io9` (missing_attribute): no golden context for this category
- `nf_clinical_study_details_20x_plaque` (missing_attribute): no golden context for this category
- `nf_ipx_rating_io9` (missing_attribute): no golden context for this category
