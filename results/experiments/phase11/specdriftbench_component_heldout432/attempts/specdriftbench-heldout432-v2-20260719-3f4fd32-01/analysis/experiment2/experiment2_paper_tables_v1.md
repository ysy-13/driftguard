# Experiment 2 paper tables

## Table E2.1 — Evidence ablation

| Evidence | Accuracy | Macro-F1 | Accuracy 95% CI | Macro-F1 95% CI | Coverage |
|---|---:|---:|---:|---:|---:|
| FIRST_FAILURE | 0.4375 | 0.3440 | [0.3889, 0.4861] | [0.2802, 0.3970] | 0.9931 |
| RETRY_HISTORY | 0.5139 | 0.4932 | [0.4583, 0.5694] | [0.4059, 0.5657] | 0.9861 |
| FULL_EVIDENCE | 0.5347 | 0.4855 | [0.4583, 0.6111] | [0.3920, 0.5716] | 0.9861 |

## Table E2.2 — Baseline comparison

| Method/View | Accuracy | Macro-P | Macro-R | Macro-F1 | Coverage |
|---|---:|---:|---:|---:|---:|
| Uniform Random Expected | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 1.0000 |
| Always-PD | 0.3333 | 0.1111 | 0.3333 | 0.1667 | 1.0000 |
| FIRST_FAILURE | 0.4375 | 0.3590 | 0.4375 | 0.3440 | 0.9931 |
| RETRY_HISTORY | 0.5139 | 0.7280 | 0.5139 | 0.4932 | 0.9861 |
| FULL_EVIDENCE | 0.5347 | 0.7482 | 0.5347 | 0.4855 | 0.9861 |
| Symbolic Oracle Upper Bound | N/A | N/A | N/A | N/A | N/A |

## Table E2.3 — Provider × evidence-view performance

| Provider | View | Accuracy | Coverage | Macro-P | Macro-R | Macro-F1 | AE Recall | TF Recall | PD Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek | FIRST_FAILURE | 0.3958 | 1.0000 | 0.3048 | 0.3958 | 0.3030 | 0.3125 | 0.0000 | 0.8750 |
| DeepSeek | RETRY_HISTORY | 0.5208 | 1.0000 | 0.7048 | 0.5208 | 0.5098 | 0.3125 | 0.3750 | 0.8750 |
| DeepSeek | FULL_EVIDENCE | 0.5208 | 1.0000 | 0.7449 | 0.5208 | 0.4537 | 0.5625 | 0.0625 | 0.9375 |
| Qwen | FIRST_FAILURE | 0.5208 | 1.0000 | 0.3852 | 0.5208 | 0.4222 | 0.6250 | 0.0000 | 0.9375 |
| Qwen | RETRY_HISTORY | 0.6250 | 1.0000 | 0.7443 | 0.6250 | 0.6246 | 0.5625 | 0.4375 | 0.8750 |
| Qwen | FULL_EVIDENCE | 0.5833 | 0.9792 | 0.7308 | 0.5833 | 0.5576 | 0.5625 | 0.2500 | 0.9375 |
| Kimi | FIRST_FAILURE | 0.3958 | 0.9792 | 0.4545 | 0.3958 | 0.2830 | 0.1875 | 0.0000 | 1.0000 |
| Kimi | RETRY_HISTORY | 0.3958 | 0.9583 | 0.4574 | 0.3958 | 0.2861 | 0.1875 | 0.0000 | 1.0000 |
| Kimi | FULL_EVIDENCE | 0.5000 | 0.9792 | 0.8034 | 0.5000 | 0.4361 | 0.4375 | 0.0625 | 1.0000 |

### Provider-specific comparison with simple baselines

| Provider | Method/View | Accuracy | Macro-F1 | Coverage |
|---|---|---:|---:|---:|
| DeepSeek | Uniform Random Expected | 0.3333 | 0.3333 | 1.0000 |
| DeepSeek | Always-PD | 0.3333 | 0.1667 | 1.0000 |
| DeepSeek | FIRST_FAILURE | 0.3958 | 0.3030 | 1.0000 |
| DeepSeek | RETRY_HISTORY | 0.5208 | 0.5098 | 1.0000 |
| DeepSeek | FULL_EVIDENCE | 0.5208 | 0.4537 | 1.0000 |
| Qwen | Uniform Random Expected | 0.3333 | 0.3333 | 1.0000 |
| Qwen | Always-PD | 0.3333 | 0.1667 | 1.0000 |
| Qwen | FIRST_FAILURE | 0.5208 | 0.4222 | 1.0000 |
| Qwen | RETRY_HISTORY | 0.6250 | 0.6246 | 1.0000 |
| Qwen | FULL_EVIDENCE | 0.5833 | 0.5576 | 0.9792 |
| Kimi | Uniform Random Expected | 0.3333 | 0.3333 | 1.0000 |
| Kimi | Always-PD | 0.3333 | 0.1667 | 1.0000 |
| Kimi | FIRST_FAILURE | 0.3958 | 0.2830 | 0.9792 |
| Kimi | RETRY_HISTORY | 0.3958 | 0.2861 | 0.9583 |
| Kimi | FULL_EVIDENCE | 0.5000 | 0.4361 | 0.9792 |

## Table E2.4 — Paired accuracy differences

| Comparison | Delta | Family-clustered 95% CI |
|---|---:|---:|
| RETRY_HISTORY - FIRST_FAILURE | 0.0764 | [0.0208, 0.1319] |
| FULL_EVIDENCE - FIRST_FAILURE | 0.0972 | [0.0417, 0.1528] |
| FULL_EVIDENCE - RETRY_HISTORY | 0.0208 | [-0.0486, 0.0903] |

Notes: all-record accuracy retains non-evaluable records in the denominator. Bootstrap unit is family (16 clusters), 10,000 repetitions, seed 20260718. Symbolic Oracle is N/A because strict protocol alignment is not proven.
