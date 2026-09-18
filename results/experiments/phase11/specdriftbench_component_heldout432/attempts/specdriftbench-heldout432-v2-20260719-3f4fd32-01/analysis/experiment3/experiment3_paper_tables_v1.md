# Experiment 3: Paper Tables

All values use the frozen formal V2 held-out Attempt.

## Table 1. False Drift Attribution

| Provider / view | AE to PD | TF to PD | False drift rate | PD recall |
|---|---:|---:|---:|---:|
| DeepSeek / FIRST_FAILURE | 11 | 14 | 0.781 | 0.875 |
| DeepSeek / RETRY_HISTORY | 11 | 10 | 0.656 | 0.875 |
| DeepSeek / FULL_EVIDENCE | 7 | 14 | 0.656 | 0.938 |
| Qwen / FIRST_FAILURE | 6 | 13 | 0.594 | 0.938 |
| Qwen / RETRY_HISTORY | 7 | 8 | 0.469 | 0.875 |
| Qwen / FULL_EVIDENCE | 7 | 8 | 0.469 | 0.938 |
| Kimi / FIRST_FAILURE | 12 | 16 | 0.875 | 1.000 |
| Kimi / RETRY_HISTORY | 11 | 16 | 0.844 | 1.000 |
| Kimi / FULL_EVIDENCE | 8 | 15 | 0.719 | 1.000 |

## Table 2. PD localization by drift type

| Type | PD recall | Category accuracy | Target accuracy | Exact-location accuracy |
|---|---:|---:|---:|---:|
| ICD | 1.000 | 1.000 | 1.000 | 0.222 |
| RSD | 1.000 | 1.000 | 1.000 | 0.444 |
| WPD | 0.750 | 0.750 | 1.000 | 0.639 |
| SED | 1.000 | 1.000 | 1.000 | 0.889 |

## Table 3. Format robustness

| Provider | First-pass valid | Repair rate | Final valid | Coverage |
|---|---:|---:|---:|---:|
| DeepSeek | 0.993 | 0.007 | 1.000 | 1.000 |
| Qwen | 0.396 | 0.604 | 0.993 | 0.993 |
| Kimi | 0.438 | 0.556 | 0.972 | 0.972 |

## Table 4. Efficiency trade-off

| Provider | Macro-F1 | Median / P95 latency ms | Tokens | Cost CNY | Cost / correct CNY |
|---|---:|---:|---:|---:|---:|
| DeepSeek | 0.435 | 3121.1 / 4015.1 | 1193726 | 1.248579 | 0.018095 |
| Qwen | 0.547 | 10235.1 / 12682.4 | 1967538 | 4.231680 | 0.050984 |
| Kimi | 0.340 | 8872.7 / 15025.2 | 1755163 | 8.298352 | 0.133844 |

## Table 5. Infrastructure reliability

| Protocol | Completed | Final infra errors | Replay | Cache miss | Ledger correct | Formal use |
|---|---:|---:|---:|---:|---|---|
| V1 | 63 / 432 | 2 | N/A | 0 | Yes | No |
| V2 | 432 / 432 | 0 | 432 / 432 | 0 | Yes | Yes |

V1 is excluded from performance tables because it is an incomplete infrastructure-gate calibration Attempt.
