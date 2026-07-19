# SpecDriftBench Held-out 432 V2 Analysis

Status: `ELIGIBLE_FORMAL_HELDOUT_RESULT`

Attempt: `specdriftbench-heldout432-v2-20260719-3f4fd32-01`

Completed: 432/432 (DeepSeek 144, Qwen 144, Kimi 144). Infrastructure errors: 0/432. Replay: 432/432 semantic matches with zero Provider calls, network calls, Cache misses, Ledger/token/cost increments, or fallback.

## Primary classification results

- Overall all-record accuracy: 49.54%
- Overall evaluable accuracy: 50.12%
- Coverage: 98.84% (427/432)
- Macro Precision / Recall / F1: 0.7233 / 0.4954 / 0.4474

Provider Macro-F1: DeepSeek 0.4352; Qwen 0.5469; Kimi 0.3396.

Evidence View Macro-F1: FIRST_FAILURE 0.3440; RETRY_HISTORY 0.4932; FULL_EVIDENCE 0.4855.

## Paired Evidence View accuracy differences

- RETRY_HISTORY - FIRST_FAILURE: +0.0764 (95% CI +0.0208, +0.1319)
- FULL_EVIDENCE - FIRST_FAILURE: +0.0972 (95% CI +0.0417, +0.1528)
- FULL_EVIDENCE - RETRY_HISTORY: +0.0208 (95% CI -0.0486, +0.0903)

Family-clustered bootstrap: 10,000 repetitions, seed 20260718, 95% percentile interval, family as the resampling unit.

## PD localization

- Category: 93.75% evaluable accuracy; 100.00% coverage.
- Target: 100.00% evaluable accuracy; 100.00% coverage.
- Location: 54.86% evaluable accuracy; 100.00% coverage.

## Quality, usage, and cost

- Schema valid: 427/432 (98.84%)
- Abstentions: 1/432 (0.23%)
- Invalid structured outputs: 4/432
- Output truncations: 1/432
- Format repairs: 168/432 (38.89%)
- Input/output tokens: 4,794,984 / 121,443
- Mean latency: 8135.89 ms; total latency: 3514705.19 ms
- Cost: CNY 13.778611094
- Leakage / fallback / main-state pollution: 0 / 0 / 0

All-record accuracy uses all 432 records; missing or invalid predictions are not counted as correct. Evaluable accuracy excludes records without an evaluable prediction. The frozen Prompt, Attribution Schema, Evidence View, Tool Registry, analysis plan, metrics, denominators, and bootstrap settings were unchanged.
