# SpecDriftBench Held-out 432 V2 Paper Tables

## Main result

| Records | Coverage | Accuracy (all) | Accuracy (evaluable) | Macro-P | Macro-R | Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 432 | 0.9884 | 0.4954 | 0.5012 | 0.7233 | 0.4954 | 0.4474 |

## Provider results

| Provider | Coverage | Accuracy (all) | Macro-F1 | Input tokens | Output tokens | Cost CNY |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek | 1.0000 | 0.4792 | 0.4352 | 1165750 | 27976 | 1.248579444 |
| Qwen | 0.9931 | 0.5764 | 0.5469 | 1918104 | 49434 | 4.231680000 |
| Kimi | 0.9722 | 0.4306 | 0.3396 | 1711130 | 44033 | 8.298351650 |

## Evidence View results

| Evidence View | Coverage | Accuracy (all) | Macro-F1 |
|---|---:|---:|---:|
| FIRST_FAILURE | 0.9931 | 0.4375 | 0.3440 |
| RETRY_HISTORY | 0.9861 | 0.5139 | 0.4932 |
| FULL_EVIDENCE | 0.9861 | 0.5347 | 0.4855 |

## Paired accuracy differences

| Comparison | Delta | 95% family-bootstrap CI |
|---|---:|---:|
| RETRY_HISTORY - FIRST_FAILURE | +0.0764 | [+0.0208, +0.1319] |
| FULL_EVIDENCE - FIRST_FAILURE | +0.0972 | [+0.0417, +0.1528] |
| FULL_EVIDENCE - RETRY_HISTORY | +0.0208 | [-0.0486, +0.0903] |

## PD metrics

| Metric | Coverage | Evaluable accuracy | All-PD accuracy |
|---|---:|---:|---:|
| Category | 1.0000 | 0.9375 | 0.9375 |
| Target | 1.0000 | 1.0000 | 1.0000 |
| Location | 1.0000 | 0.5486 | 0.5486 |

Infrastructure errors: 0. Replay semantic matches: 432/432. Paper eligibility: `ELIGIBLE_FORMAL_HELDOUT_RESULT`.
