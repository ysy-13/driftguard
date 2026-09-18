# SpecDriftBench Protocol-Rule Baseline

Status: `POST_HOC_CAMERA_READY_BASELINE`.

The classifier consumes only the same Agent-visible evidence supplied to the LLMs. It makes no model or network calls and has no family- or provider-specific rules.

## Rules

1. visible call or response interpretation violates the displayed contract -> AGENT_ERROR
2. exact legal retry recovers -> TRANSIENT_FAILURE
3. exact legal retry reproduces the failure -> PERSISTENT_DRIFT
4. otherwise, a conforming first failure uses a declared conservative PERSISTENT_DRIFT fallback

## Held-out results (144 unique evidence inputs)

| View | Accuracy | Macro-F1 | Coverage |
|---|---:|---:|---:|
| FIRST_FAILURE | 0.6667 | 0.5556 | 1.0000 |
| RETRY_HISTORY | 1.0000 | 1.0000 | 1.0000 |
| FULL_EVIDENCE | 1.0000 | 1.0000 | 1.0000 |
| Overall | 0.8889 | 0.8857 | 1.0000 |

The provider-aligned 432-row expansion repeats each deterministic prediction in the three provider slots. It has identical rates and does not create additional independent observations.

## Limitations

- This baseline was added after peer review and was not preregistered with the original analysis plan.
- The conservative first-failure fallback is a declared tie-breaking policy, not evidence that drift is present.
- The rule baseline evaluates causal classification only; it does not provide drift-category or location predictions.
