# SpecDriftBench Held-out 432 Protocol Amendment 1.1

Status: `FROZEN_BEFORE_V2_REAL_RESULTS`

Infrastructure Gate Version: `2`

## Reason for the amendment

The V1 Attempt `specdriftbench-heldout432-20260718-af3db03-01` stopped after
63/432 records. DeepSeek produced two non-consecutive final Provider errors in
63 completed records (2/63), with six successful records between them. The
persisted errors do not contain an HTTP status, Provider code, request ID, or
underlying `httpx` exception class, so their exact root cause is
`ROOT_CAUSE_METADATA_INSUFFICIENT`. The V1 cumulative-two-errors rule was too
strict for a 144-record Provider stage.

The V1 Attempt is permanently classified
`INCOMPLETE_INFRASTRUCTURE_GATE_CALIBRATION_NOT_FOR_PAPER`. Its 63 records are
operational diagnostics only. They cannot enter paper tables, be merged with a
V2 Attempt, fill V2 missing records, or be reused as formal V2 results.

## Frozen V2 change

V2 changes only Provider transport classification, error observability, and
the experiment-level infrastructure aggregation gate. It does not change the
Prompt, Attribution Schema, Evidence View, Tool Registry, benchmark split,
ground truth, Provider order, model IDs or parameters, format-repair limit,
analysis plan, metrics, denominators, budgets, or 432-record paired design.
No V1 accuracy or Evidence View result was used to choose the amendment.

After the frozen Provider retry policy is exhausted, a Provider stage stops
when any of the following occurs:

- three consecutive records end in a final infrastructure error;
- at least 24 Provider records are complete and final infrastructure errors
  divided by completed Provider records is at least 10%;
- a systemic permanent authentication, balance, model, or endpoint error is
  identified.

A success resets the consecutive counter. Unknown Provider client errors enter
the aggregate infrastructure denominator. A request-level 400, 413, 415, or
422 is retained as a non-evaluable record but does not independently stop the
Attempt. HTTP 401, 402, invalid API key, insufficient balance, model-not-found,
and endpoint configuration failures stop immediately.

The retry policy remains bounded to the initial call plus the configured three
Provider retries. DNS, connection, TLS, timeout, HTTP 408/429, and HTTP
500/502/503/504 are retryable. Permanent 4xx responses are not retryable.

## Isolation and authorization

V2 starts from a new 432-record Attempt with new result, Cache, and Checkpoint
namespaces. The incomplete V1 Attempt is explicitly excluded. Its Cache and
Checkpoints cannot be resumed or reused. A V2 real run still requires all
three gates: an authorized temporary V2 config, `--allow-real-api`, and the
exact confirmation token `RUN-EXACTLY-432-V2`. The V1 token cannot authorize
V2, and the V2 token cannot authorize V1.

The thresholds in this amendment are frozen before any V2 real result exists.
