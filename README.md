# DriftGuard

This repository contains phase 1 of the **DriftGuard** benchmark. Its canonical
OpenAPI 3.1 contract is
[`benchmark/openapi/driftguard_openapi_v1.yaml`](benchmark/openapi/driftguard_openapi_v1.yaml).

At experiment initialization, both `displayed_spec` (the contract shown to an
agent) and `runtime_contract` (the contract enforced by the API) are generated
from this canonical v1 file, so they begin identical. The canonical file must
never contain an injected drift. Future drift injection must copy this file and
apply a patch to the copy rather than mutating the canonical source.

## Setup and validation

```bash
python -m pip install -r requirements.txt
python scripts/validate_openapi.py
pytest -q
```

The validator checks formal OpenAPI 3.1 validity plus DriftGuard-specific
invariants for the 12 canonical tools, internal references, metadata, inputs,
outputs, and known drift exclusions.

## Phase 1 scope

Phase 1 intentionally did **not** implement task templates, initial sandbox
state, drift cases, matched failures, API handlers, persistence, diagnosis,
LLM/framework integration, contract mutation, or a frontend. It defines and
tests only the reusable canonical v1 tool contract.

## Phase 2: Canonical Tasks

Phase 2 adds the deterministic fixture
[`benchmark/fixtures/initial_state_v1.json`](benchmark/fixtures/initial_state_v1.json)
and 32 canonical tasks in
[`benchmark/tasks/tasks_v1.json`](benchmark/tasks/tasks_v1.json). `S0` is the
fixed initial sandbox state containing repositories, users, members, issues,
workflows, and pipeline runs. Every task uses `reset_policy: isolated`: it starts
from a deep copy of S0 and discards state changes after scoring.

The tasks are divided into 13 calibration, 11 detection, and 8 future-transfer
examples. An `oracle_plan` is a shortest reference trace, not the only accepted
trace; later evaluation should primarily use `success_assertions` for the target
state and `forbidden_assertions` to reject unrelated writes. Extra safe reads
may therefore be acceptable.

This phase defines data and assertions only. It does not execute tools, call an
LLM, score natural-language answers, inject drift, or create matched failures.
Those capabilities belong to later phases.

Run the complete validation suite with:

```bash
python scripts/validate_openapi.py
python scripts/validate_tasks.py
pytest -q
```

## Phase 3: Persistent Drift Cases

Phase 3 defines 20 persistent tool-specification drift cases in
[`benchmark/drifts/drift_cases_v1.json`](benchmark/drifts/drift_cases_v1.json):
five input-contract, five response-shape, five workflow-precondition, and five
state-effect cases. Their task coverage, independent detection instances,
held-out future-transfer instances, and regression mappings are stored in
[`benchmark/drifts/task_drift_coverage_v1.json`](benchmark/drifts/task_drift_coverage_v1.json).

The agent-visible displayed specification remains canonical v1. Each runtime
mutation is only a declarative change intended for a future copy of the runtime
contract; this phase does not mutate or execute the canonical source. One
failure is insufficient evidence: every case requires at least two independent
failures, excludes transient 429/503 evidence, requires a probe for a candidate
patch, and requires at least three regression tasks. Future-transfer instances
are held out and cannot be used to generate patches.

This phase still contains data definitions and validation only. It does not
execute drift, implement matched failures, or introduce Agent Error and
Transient Failure conditions; those belong to Phase 4.

Run all three phase validators and tests with:

```bash
python scripts/validate_openapi.py
python scripts/validate_tasks.py
python scripts/validate_drifts.py
pytest -q
```

## Phase 4: Matched Failure Attribution

Phase 4 defines matched failure triplets in
[`benchmark/matched_failures/matched_failures_v1.json`](benchmark/matched_failures/matched_failures_v1.json).
Each of the 20 persistent drift families has three root-cause variants—Agent
Error, Transient Failure, and Persistent Drift—for 60 scenarios and 360
deterministic episodes. The variants share a normalized first-failure symptom
so attribution must use contract alignment, generated calls, recurrence, state
diffs, probes, and history rather than an obvious error-message clue.

The six-episode protocol supplies two baselines, a matched first failure,
replication or recovery, probe or confirmation, and held-out transfer. Ground
truth is evaluator-only; diagnoser-visible evidence bundles cannot contain
labels, variant codes, expected actions, or patch permissions. Only validated
Persistent Drift permits a permanent patch. A patch proposed for Agent Error or
Transient Failure counts as a false patch.

This phase defines scenarios and evaluator metadata only. It does not execute
tools or faults, run an Agent or classifier, generate patches, or perform shadow
validation.

Run all four phase validators and tests with:

```bash
python scripts/validate_openapi.py
python scripts/validate_tasks.py
python scripts/validate_drifts.py
python scripts/validate_matched_failures.py
pytest -q
```

## Phase 5: Deterministic Canonical Sandbox

Phase 5 adds the first executable canonical environment. A `StateStore` creates
an isolated deep copy of S0 for every task, while `DeterministicClock` advances
exactly one second after each successful write and resets with the fixture. The
canonical OpenAPI is loaded into a contract registry that supplies flattened
tool inputs, permissions, defaults, effect metadata, and success schemas to the
same `SandboxService` used by both Oracle execution and HTTP requests.

All 12 canonical tool handlers implement repository, issue, pipeline, and
membership behavior without any drift. Calls produce immutable-style results,
pre/post snapshots, deterministic JSON-Pointer state diffs, and audit records
outside business state. The data-driven Oracle runner resolves step bindings
and conditions, evaluates state/answer/forbidden assertions, and runs every one
of the 32 tasks independently from S0 with deterministic replay.

FastAPI is only a thin local experiment adapter: its 12 public method/path and
operation IDs come from the same registry and delegate directly to
`SandboxService.call_tool()`. `/internal/reset` is hidden from its OpenAPI. This
phase has no drift injection, LLM, Agent, classifier, or patch execution.

Install and run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts/validate_openapi.py
python scripts/validate_tasks.py
python scripts/validate_drifts.py
python scripts/validate_matched_failures.py
python scripts/run_oracle.py
pytest -q
```

Optionally start the local-only API sandbox:

```bash
uvicorn driftguard.api.app:app --reload
```

## Phase 6: Executable Failure Injection

Phase 6 adds an optional execution context around the unchanged canonical
handlers. Runtime profiles select Agent Error (one episode), Transient Failure
(one runtime occurrence), or Persistent Drift (from its change point until a
candidate patch is active). Contract snapshots are isolated deep copies;
displayed and runtime overlays never modify the canonical OpenAPI file.

Registered mutation strategies cover all 20 input-contract, response-shape,
workflow-precondition, and state-effect cases. Deterministic workflow
verification tokens and deferred state effects live only in the scenario
session sidecar; pending effects materialize on their specified read and never
enter ordinary business-state diffs. Raw canonical and Agent-visible runtime
responses remain separately auditable. The observation normalizer maps
executable results to the 20 matched family signatures without exposing
ground-truth labels.

Run the Phase 6 first-failure conformance check with:

```bash
python scripts/validate_openapi.py
python scripts/validate_tasks.py
python scripts/validate_drifts.py
python scripts/validate_matched_failures.py
python scripts/run_oracle.py
python scripts/run_injection_conformance.py
pytest -q
```

It executes 60 matched first failures, verifies AE/TF recovery and PD
persistence on independent calls, checks family-level symptom equality, and
rechecks SHA-256 hashes for all six protected canonical benchmark files. It
does not run an LLM, agent, patch generator, shadow validator, or the complete
360-episode experiment.
