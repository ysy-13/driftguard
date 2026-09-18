# Experiment 3: Provider Efficiency

Attempt: `specdriftbench-heldout432-v2-20260719-3f4fd32-01`

| Provider | Accuracy | Evaluable accuracy | Macro-F1 | Coverage | Input tokens | Output tokens | Median ms | P95 ms | Network attempts | Repairs | Cost CNY |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek | 0.479 | 0.479 | 0.435 | 1.000 | 1165750 | 27976 | 3121.1 | 4015.1 | 150 | 1 | 1.248579 |
| Qwen | 0.576 | 0.580 | 0.547 | 0.993 | 1918104 | 49434 | 10235.1 | 12682.4 | 231 | 87 | 4.231680 |
| Kimi | 0.431 | 0.443 | 0.340 | 0.972 | 1711130 | 44033 | 8872.7 | 15025.2 | 224 | 80 | 8.298352 |

Latency audit: `HOST_SUSPENSION_OR_NETWORK_DELAY_CANNOT_BE_DISAMBIGUATED`.

Provider strata are descriptive outcomes under the frozen configuration; no causal or universal best-provider claim is made.
