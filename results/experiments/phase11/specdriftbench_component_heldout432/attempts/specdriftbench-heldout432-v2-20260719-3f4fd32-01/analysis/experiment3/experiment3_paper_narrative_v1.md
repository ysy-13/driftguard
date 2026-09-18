# Experiment 3: Paper Narrative

The formal V2 held-out analysis indicates that the three evaluated LLM providers systematically over-attributed non-drift failures to persistent drift within this benchmark: 80 of 144 agent-error records and 114 of 144 transient-failure records were classified as PD. TF was the hardest class to identify, with lower recall than AE and PD. This is a descriptive result for the frozen synthetic benchmark and provider configurations, not evidence of a general model property or all real API evolution.

Evidence history was associated with asymmetric class changes. Retry history improved transient-failure recall relative to first-failure evidence, while full evidence recovered some agent-error decisions and gave back part of the transient-failure gain. The paired transition counts show both corrections and regressions, so the result does not support a monotonic "more evidence is always better" claim.

Among the 144 actual PD records, category accuracy was 0.938, target accuracy was 1.000, and exact-location accuracy was 0.549. These are separately scored PD-only components; target accuracy exceeding category accuracy is therefore not a contradiction and should not be presented as a monotone funnel.

Exact localization was the narrowest component. The error taxonomy distinguishes wrong-field, overly broad or narrow paths, wrong component or layer, wrong category, missing location, and non-evaluable outputs without changing the frozen evaluator's scoring rule.

Format robustness varied substantially by provider: the frozen run recorded 1 DeepSeek repair, 87 Qwen repairs, and 80 Kimi repairs. Of 168 total repair paths, 164 ended in schema-valid output. The most format-stable provider was not the most diagnostically accurate provider, indicating that format stability and diagnostic accuracy are distinct measured capabilities. This supports reporting first-pass and final validity separately rather than treating repaired outputs as first-pass successes.

Repairs introduced measurable additional tokens, latency, and estimated cost. Semantic-change rates are conditioned only on repaired responses with comparable initial core-class fields; invalid or missing-core-field initial outputs are explicitly excluded from that denominator.

Provider trade-offs were not one-dimensional. Estimated costs were CNY 1.248579 for DeepSeek, 4.231680 for Qwen, and 8.298352 for Kimi, alongside different Macro-F1, latency, coverage, and first-pass format stability. The frozen observations do not establish causality or a universally best provider.

The earlier V1 Attempt completed only 63 of 432 records and stopped after two final infrastructure errors under the original gate. It is classified as `INCOMPLETE_INFRASTRUCTURE_GATE_CALIBRATION_NOT_FOR_PAPER` and is not used for performance comparison. V2 completed 432 records with zero final infrastructure errors and reproduced all 432 results in an offline replay with zero Cache misses. V2 improved only the experiment's operational reliability; it did not improve the models themselves.
