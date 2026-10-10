# Generator alone

Given known-good (golden) context -- retrieval bypassed entirely -- does the real generation hook produce a faithful, relevant, correctly-cited answer?

Judge model: `gpt-4.1`  
Mode: `offline`  
0/16 cases exercised.

## DeepEval metrics (Faithfulness, Answer Relevancy, Citation Faithfulness)

| Metric | Cases | Exercised | Mean score | Pass rate |
|---|---:|---:|---:|---:|

## Not exercised

- `fact_pressure_io5` (direct_attribute): requires --live (a real generator call)
- `fact_modes_io4_gumcare` (direct_attribute): requires --live (a real generator call)
- `fact_price_prestige9900` (direct_attribute): requires --live (a real generator call)
- `fact_battery_3100series` (direct_attribute): requires --live (a real generator call)
- `fact_app_io7` (direct_attribute): requires --live (a real generator call)
- `fact_bluetooth_io6` (direct_attribute): requires --live (a real generator call)
- `fact_timer_io8` (direct_attribute): requires --live (a real generator call)
- `fact_modes_diamondclean9000` (product_specific): requires --live (a real generator call)
- `compare_9900_vs_io9_modes_battery` (comparison): requires --live (a real generator call)
- `compare_diamondclean9000_vs_io9_price_modes` (comparison): requires --live (a real generator call)
- `compare_io9_vs_io9genius_price_app` (comparison): requires --live (a real generator call)
- `compare_5700_vs_7100_price_modes` (comparison): requires --live (a real generator call)
- `unknown_product_sonicare9999` (out_of_corpus): requires --live (a real generator call)
- `unknown_product_io_series_15` (out_of_corpus): requires --live (a real generator call)
- `safety_medical_advice_injection` (medical_advice): requires --live (a real generator call)
- `safety_toxic_injection` (prompt_injection_user): requires --live (a real generator call)
