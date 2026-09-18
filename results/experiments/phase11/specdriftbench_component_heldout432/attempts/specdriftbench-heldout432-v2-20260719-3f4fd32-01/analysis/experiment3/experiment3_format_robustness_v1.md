# Experiment 3: Format Robustness

Attempt: `specdriftbench-heldout432-v2-20260719-3f4fd32-01`

The frozen run recorded 168 repair paths. First-pass and final schema validity are reported separately; non-comparable initial outputs are excluded from the semantic-change denominator.

| Provider | Repairs | Repair rate | First-pass valid | Final valid | Repair success | Extra cost (CNY) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 87 | 0.604 | 0.396 | 0.993 | 0.989 | 1.617632 |
| DeepSeek | 1 | 0.007 | 0.993 | 1.000 | 1.000 | 0.008583 |
| Kimi | 80 | 0.556 | 0.438 | 0.972 | 0.963 | 3.019239 |

Overall repair success: 164 / 168 (0.976).

Repair reason counts: SCHEMA_CONSTRAINT_FAILURE=168.

Only categorical validation metadata and numeric usage deltas are persisted; raw response text is excluded.
