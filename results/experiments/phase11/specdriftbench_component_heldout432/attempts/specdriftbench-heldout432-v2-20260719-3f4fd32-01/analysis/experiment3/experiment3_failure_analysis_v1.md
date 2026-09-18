# Experiment 3: Failure Analysis

Attempt: `specdriftbench-heldout432-v2-20260719-3f4fd32-01`

Status: formal V2 held-out offline analysis.

## False drift attribution

False Drift Attribution Rate is predicted PD among actual AE/TF divided by actual AE/TF records. The evaluable-subset rate uses only schema-valid class predictions.

| Scope | Records | Evaluable | Predicted PD | All-record rate | Evaluable rate |
|---|---:|---:|---:|---:|---:|
| Overall | 288 | 283 | 194 | 0.674 | 0.686 |
| FIRST_FAILURE | 96 | 95 | 72 | 0.750 | 0.758 |
| FULL_EVIDENCE | 96 | 94 | 59 | 0.615 | 0.628 |
| RETRY_HISTORY | 96 | 94 | 63 | 0.656 | 0.670 |

Observed confusion counts: AE_to_PD=80, TF_to_PD=114, PD_to_PD=135.

## Persistent-drift localization

Category, target, and exact-location values are separately scored PD-only component accuracies; they are not assumed to form a monotone funnel. Strict nested survival is included only as a descriptive supplement.

| Component | Correct / PD | Coverage | All-record accuracy | Evaluable accuracy |
|---|---:|---:|---:|---:|
| class | 135 / 144 | 1.000 | 0.938 | 0.938 |
| category | 135 / 144 | 1.000 | 0.938 | 0.938 |
| target | 144 / 144 | 1.000 | 1.000 | 1.000 |
| location | 79 / 144 | 1.000 | 0.549 | 0.549 |

## Infrastructure V1/V2

V1 is retained only as an incomplete infrastructure-gate calibration trace. The completed V2 attempt is the sole performance source; this comparison supports no model-performance claim.

V1 performance metrics are excluded.
