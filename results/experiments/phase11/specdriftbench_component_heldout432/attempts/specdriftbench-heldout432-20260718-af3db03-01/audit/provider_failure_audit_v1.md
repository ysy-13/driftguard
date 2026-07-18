# SpecDriftBench Held-out 432 V1 Provider Failure Audit

Status: `INCOMPLETE_INFRASTRUCTURE_GATE_CALIBRATION_NOT_FOR_PAPER`

This is an offline, read-only audit of Attempt
`specdriftbench-heldout432-20260718-af3db03-01`. No Provider was called, and
the Attempt was not resumed or replayed.

## Finding

Both failed records persisted the same sanitized signature:
`PROVIDER_ERROR`, status `null`, code `null`, request ID `null`, failure layer
`UNKNOWN`, message `provider HTTP client error`, and `retryable=false`. Each
made one logical boundary call and one network attempt, with no Provider retry,
format repair, usage, Cache entry, or persisted response metadata. Both wrote a
Checkpoint, and all Ledger reservations were released (`reserved_cny=0`).

The first error was record `c97d503a9c6d6d4433dc0d5c`, DeepSeek/M09/AE
under `RETRY_HISTORY` (`public-4bf040e046e4`, ordinal 56). Ordinal 55
succeeded, and ordinal 57 succeeded immediately afterward.

The second error was record `1f0bd7f850eec55163ac969a`, DeepSeek/M09/PD
under `FULL_EVIDENCE` (`public-9c6340cc26ca`, ordinal 63). Ordinal 62
succeeded; the V1 aggregate gate then stopped the Attempt.

The errors were not consecutive: six successful records (57–62) separated
them. The count reconciles exactly as 61 successful first responses + 2 failed
initial calls + 2 format-repair calls = 65 Ledger/API attempts.

## Root cause and retry decision

The code path is provable: a residual `httpx.HTTPError` was caught and converted
to `ProviderError(retryable=false, failure_layer=UNKNOWN)`, so the retry loop
raised it after the first attempt. The underlying exception class and any HTTP
response were not persisted. Therefore the precise root cause is
`ROOT_CAUSE_METADATA_INSUFFICIENT`, and the operational classification is
`UNKNOWN_HTTP_CLIENT_ERROR`.

There is no persisted evidence of HTTP 401/402, invalid credentials,
insufficient balance, model/endpoint failure, global Prompt/Schema/Evidence
incompatibility, or leakage. Sixty-one of 63 records completed successfully.
The V1 rule that stopped a 144-record Provider stage after any two
infrastructure errors was therefore overly strict; this conclusion uses only
transport reliability evidence, not model accuracy.

## Permanent disposition

The 63 records are operational diagnostics only. They must not enter paper main
tables, be merged with a future Attempt, fill missing future records, or be
reused as formal V2 results. The complete V1 Attempt, including its original
records, Cache, Checkpoints, Manifest, Summary, Provider gate, Ledger linkage,
and partial analysis, must remain preserved for audit.

Pre-audit hashes:

- Existing Attempt tree: `a6a01af230f9beb70ac581d70a250323abb4280a7625f85064a7f093e0b7bb5d`
- Records: `253d6a2fc10ebd9d861c10a0aa72baebf72d5e87886d7a537db91ddb822d2c2f`
- Checkpoints: `7adc96852cb835ef1cb9b14e12fc4464a18ed26ea403e67b7485e6063ebd9c5e`
- Cache: `10ab0a90aab3e9e20350c781d1ba2aed33a65e64ed25ff4d544957736bbbf436`
