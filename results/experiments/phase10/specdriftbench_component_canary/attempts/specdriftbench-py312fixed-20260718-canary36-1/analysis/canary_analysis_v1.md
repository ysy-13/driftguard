# SpecDriftBench Component Canary Analysis v1

- Attempt: `specdriftbench-py312fixed-20260718-canary36-1`
- Status: `VALID_DEVELOPMENT_CANARY`
- Freeze gate: `READY_TO_FREEZE_FOR_HELDOUT`
- Scope: 36 development Canary records; 0 held-out records.
- Analysis source: existing normalized records plus evaluator ground truth only; no API calls, raw-response reading, output rescoring, or significance tests.
- Evidence View results: `DEVELOPMENT_CANARY_DESCRIPTIVE_ONLY`.

## Replay and integrity

Replay completed 36/36 with 0 Cache misses, 0 Provider instances, 0 network calls, and zero Ledger attempts/token/cost deltas. All 31 compared semantic fields matched, including the 6 format-repair paths and original Cache references. Ledger SHA-256 remained byte-identical.

The prior attempt `specdriftbench-k26nt1-20260717-canary36-6-a91c7e4b` is referenced as `TRANSPORT_INVALID_OLD_LIBRESSL_ENVIRONMENT`; its files were not modified.

## Overall classification

| Ground truth | Predicted AE | Predicted TF | Predicted PD |
|---|---:|---:|---:|
| AE | 7 | 0 | 5 |
| TF | 0 | 0 | 12 |
| PD | 0 | 0 | 12 |

- All-record accuracy: 52.78% (19/36)
- Evaluable-subset accuracy: 52.78% (19/36)
- Coverage/evaluable rate: 100.00% (36/36)
- Macro precision: 0.471264 (mean over 3 classes)
- Macro recall: 0.527778 (mean over 3 classes)
- Macro F1: 0.440736 (mean over 3 classes)

## Per-class metrics

| Variant | Class | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| AE | AGENT_ERROR | 1.000000 (7/7) | 0.583333 (7/12) | 0.736842 |
| TF | TRANSIENT_FAILURE | 0.000000 (0/0) | 0.000000 (0/12) | 0.000000 |
| PD | PERSISTENT_DRIFT | 0.413793 (12/29) | 1.000000 (12/12) | 0.585366 |

## Provider metrics

| Provider | Accuracy | Coverage | Macro-F1 | Repairs | Input tokens | Output tokens | Latency | Cost (CNY) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek | 50.00% (6/12) | 100.00% (12/12) | 0.412698 | 0 | 97671 | 2199 | 32.735s | 0.104314518 |
| dashscope | 58.33% (7/12) | 100.00% (12/12) | 0.490842 | 4 | 131692 | 3582 | 87.729s | 0.292040000 |
| moonshot | 50.00% (6/12) | 100.00% (12/12) | 0.412698 | 2 | 106232 | 2251 | 85.899s | 0.506376910 |

## Evidence View metrics

### FIRST_FAILURE

Interpretation: `DEVELOPMENT_CANARY_DESCRIPTIVE_ONLY`.

| Ground truth | Predicted AE | Predicted TF | Predicted PD |
|---|---:|---:|---:|
| AE | 3 | 0 | 3 |
| TF | 0 | 0 | 3 |
| PD | 0 | 0 | 3 |

Accuracy: 50.00% (6/12); coverage: 100.00% (12/12); Macro-F1: 0.388889.

### RETRY_HISTORY

Interpretation: `DEVELOPMENT_CANARY_DESCRIPTIVE_ONLY`.

| Ground truth | Predicted AE | Predicted TF | Predicted PD |
|---|---:|---:|---:|
| AE | 1 | 0 | 2 |
| TF | 0 | 0 | 6 |
| PD | 0 | 0 | 3 |

Accuracy: 33.33% (4/12); coverage: 100.00% (12/12); Macro-F1: 0.309524.

### FULL_EVIDENCE

Interpretation: `DEVELOPMENT_CANARY_DESCRIPTIVE_ONLY`.

| Ground truth | Predicted AE | Predicted TF | Predicted PD |
|---|---:|---:|---:|
| AE | 3 | 0 | 0 |
| TF | 0 | 0 | 3 |
| PD | 0 | 0 | 6 |

Accuracy: 75.00% (9/12); coverage: 100.00% (12/12); Macro-F1: 0.600000.

## PD attribution

- PD category accuracy: 100.00% (12/12)
- PD target accuracy: 100.00% (12/12)
- PD location accuracy: 58.33% (7/12)
- Each metric separately reports all-record accuracy, evaluable-subset accuracy, and coverage in the JSON artifact. For example, PD target accuracy is correct PD target / evaluable PD target records.

## Output quality, safety, and usage

- Schema-valid rate: 100.00% (36/36)
- Abstention rate: 0.00% (0/36)
- Final invalid structured-output rate: 0.00% (0/36)
- Format-repair rate: 16.67% (6/36); 6 repairs total (deepseek 0, dashscope 4, moonshot 2).
- Infrastructure error rate: 0.00% (0/36)
- API Key leakage rate: 0.00% (0/36)
- Ground-truth leakage rate: 0.00% (0/36)
- Usage: 335595 input tokens, 8032 output tokens, 42 Provider attempts, 206.363s aggregate latency, CNY 0.902731428.

## Denominator policy and limitations

All-record accuracy uses every record and never treats missing predictions as correct. Evaluable-subset accuracy uses only records with the relevant evaluable prediction. Coverage is evaluable records / all records in the stated slice. Zero-denominator class precision, recall, and F1 are reported as 0.

These 36 development Canary observations are descriptive only. No significance testing or held-out inference is permitted, and the results must not be used to modify the Prompt, Schema, Evidence View, benchmark split, or model configuration.

## Freeze check

Full test suite: 720 passed, 0 failed, 1 non-blocking deprecation warning. `git diff --check` passed. All held-out gate conditions recorded in the JSON artifact are true.

`READY_TO_FREEZE_FOR_HELDOUT`

No held-out run was performed.
