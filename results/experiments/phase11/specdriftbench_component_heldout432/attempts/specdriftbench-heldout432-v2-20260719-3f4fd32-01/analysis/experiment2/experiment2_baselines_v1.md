# Experiment 2 baselines

| Baseline | Accuracy | Macro-P | Macro-R | Macro-F1 | Coverage | Classification |
|---|---:|---:|---:|---:|---:|---|
| Uniform Random Expected | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 1.0000 | ANALYTICAL_EXPECTED_BASELINE |
| Always-PD | 0.3333 | 0.1111 | 0.3333 | 0.1667 | 1.0000 | DETERMINISTIC_BASELINE |
| Symbolic Oracle Upper Bound | N/A | N/A | N/A | N/A | N/A | NON_COMPARABLE_SYMBOLIC_ORACLE_UPPER_BOUND |

Uniform Random is a closed-form expectation and is not an API run. Always-PD is recomputed programmatically across all 432 records. Symbolic Oracle is N/A because protocol alignment is not proven: the Phase 7 result uses benchmark-specific deterministic rules, is not a fair LLM baseline, cannot participate in model ranking, and is only eligible as a solvability check after strict alignment.
