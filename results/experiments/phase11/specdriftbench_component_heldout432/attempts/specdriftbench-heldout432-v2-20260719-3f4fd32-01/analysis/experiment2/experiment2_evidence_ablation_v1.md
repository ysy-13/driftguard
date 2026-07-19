# Experiment 2: Evidence Ablation and Baseline Comparison

- Attempt: `specdriftbench-heldout432-v2-20260719-3f4fd32-01`
- Scope: frozen V2 held-out records only; fully offline
- Records: 432 across 16 held-out families
- Bootstrap: family-clustered, 10,000 repetitions, seed 20260718, percentile 95% CI

## Evidence-view performance

| Evidence view | Accuracy (all) | Accuracy (evaluable) | Coverage | Macro-P | Macro-R | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| FIRST_FAILURE | 0.4375 | 0.4406 | 0.9931 | 0.3590 | 0.4375 | 0.3440 |
| RETRY_HISTORY | 0.5139 | 0.5211 | 0.9861 | 0.7280 | 0.5139 | 0.4932 |
| FULL_EVIDENCE | 0.5347 | 0.5423 | 0.9861 | 0.7482 | 0.5347 | 0.4855 |

## Paired evidence gains

| Comparison | All-record delta | 95% CI | Joint-evaluable delta | Coverage delta | A+/B- | A-/B+ |
|---|---:|---:|---:|---:|---:|---:|
| RETRY_HISTORY - FIRST_FAILURE | 0.0764 | [0.0208, 0.1319] | 0.0851 | -0.0069 | 17 | 6 |
| FULL_EVIDENCE - FIRST_FAILURE | 0.0972 | [0.0417, 0.1528] | 0.0922 | -0.0069 | 16 | 2 |
| FULL_EVIDENCE - RETRY_HISTORY | 0.0208 | [-0.0486, 0.0903] | 0.0213 | 0.0000 | 13 | 10 |

The Retry–First and Full–First intervals exclude zero. The Full–Retry interval crosses zero, so the formal data do not support a clear additional accuracy gain from full evidence over retry history. Retry history captures most of the observed gain.

## Baselines

| Baseline | Accuracy | Macro-P | Macro-R | Macro-F1 | Status |
|---|---:|---:|---:|---:|---|
| Uniform random (analytical expectation) | 0.3333 | 0.3333 | 0.3333 | 0.3333 | ANALYTICAL_EXPECTED_BASELINE |
| Always-PD | 0.3333 | 0.1111 | 0.3333 | 0.1667 | DETERMINISTIC_BASELINE |
| Symbolic Oracle | N/A | N/A | N/A | N/A | NON_COMPARABLE_SYMBOLIC_ORACLE_UPPER_BOUND: protocol alignment not proven |

FIRST_FAILURE Macro-F1 is only slightly above the analytical random expectation (0.3440 vs 0.3333). The richer views improve class balance, but persistent-drift overprediction remains the dominant error pattern.

## Class-level mechanism

| Actual class | View | Correct | Incorrect | Non-evaluable | Recall | Coverage |
|---|---|---:|---:|---:|---:|---:|
| AGENT_ERROR | FIRST_FAILURE | 18 | 29 | 1 | 0.3750 | 0.9792 |
| AGENT_ERROR | RETRY_HISTORY | 17 | 29 | 2 | 0.3542 | 0.9583 |
| AGENT_ERROR | FULL_EVIDENCE | 25 | 22 | 1 | 0.5208 | 0.9792 |
| TRANSIENT_FAILURE | FIRST_FAILURE | 0 | 48 | 0 | 0.0000 | 1.0000 |
| TRANSIENT_FAILURE | RETRY_HISTORY | 13 | 35 | 0 | 0.2708 | 1.0000 |
| TRANSIENT_FAILURE | FULL_EVIDENCE | 6 | 41 | 1 | 0.1250 | 0.9792 |
| PERSISTENT_DRIFT | FIRST_FAILURE | 45 | 3 | 0 | 0.9375 | 1.0000 |
| PERSISTENT_DRIFT | RETRY_HISTORY | 44 | 4 | 0 | 0.9167 | 1.0000 |
| PERSISTENT_DRIFT | FULL_EVIDENCE | 46 | 2 | 0 | 0.9583 | 1.0000 |

Persistent-drift predictions remain concentrated across views. Retry evidence primarily restores transient-failure recall, while full evidence raises agent-error recall but gives back part of the transient-failure gain; this explains why accuracy and Macro-F1 order the two richest views differently.

## Validity and use

This is a confirmatory offline analysis of the frozen formal V2 held-out Attempt. V1 is excluded as `INCOMPLETE_INFRASTRUCTURE_GATE_CALIBRATION_NOT_FOR_PAPER`. No API, provider, cache, credential, prompt, schema, evidence, benchmark, or model path was used or changed. The results are suitable for the paper as evidence-ablation and deterministic/analytical baseline comparisons, subject to the stated denominator and oracle non-comparability notes.
