# DriftGuard / SpecDriftBench

This repository contains the implementation and released research artifact for
**SpecDriftBench**. For the paper-specific artifact inventory, frozen formal
results, and offline reproduction instructions, see [`ARTIFACT.md`](ARTIFACT.md).

Its canonical
OpenAPI 3.1 contract is
[`benchmark/openapi/driftguard_openapi_v1.yaml`](benchmark/openapi/driftguard_openapi_v1.yaml).

The code and released artifact are available under the
[`MIT License`](LICENSE).

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
rechecks SHA-256 hashes for all seven protected canonical benchmark files. It
does not run an LLM, agent, patch generator, shadow validator, or the complete
360-episode experiment.

## Phase 7: Evidence-Based Failure Attribution

Phase 7 adds immutable, append-only evidence traces and strictly separates the
Agent-visible view from evaluator-only ground truth. Scenario-local history is
time bounded, while recursive leakage checks reject hidden runtime profiles,
variant labels, drift identifiers, evaluator metadata, and future evidence.

The deterministic diagnosis engine uses displayed-spec validation, visible
responses, state observations, prior successes, controlled retries, and safe
active probes to distinguish Agent Error, Transient Failure, and Persistent
Spec Drift. Generic localization rules identify ICD, RSD, WPD, and SED
locations from the displayed OpenAPI and observed symptom; they do not use
family IDs or source drift IDs. The patch eligibility gate only reports
whether the attribution protocol is satisfied—it does not generate or apply a
patch.

All write-capable probes execute on isolated state forks. Probe budgets,
before/after hashes, unsafe-probe rejection, deterministic replay, Oracle
regression, Phase 6 regression, and canonical hashes are included in the
attribution report.

Run:

```bash
python scripts/run_attribution_conformance.py
python scripts/run_attribution_conformance.py --family M01
python scripts/run_oracle.py
python scripts/run_injection_conformance.py
pytest -q
```

This phase uses no external LLM API and does not generate patches, run Phase 8
self-repair, or modify the canonical handlers.

## Phase 8: Evidence-Grounded Tool-Spec Healing

Phase 8 turns an eligible Phase 7 Persistent Drift diagnosis into a minimal,
auditable `ToolSpecPatch`. The deterministic candidate generator receives only
the frozen `AgentView`, `DiagnosisResult`, exact localization, eligibility
decision, displayed OpenAPI snapshot, and cited visible evidence. It cannot
accept an `EvaluatorView` or `RuntimeProfile`, and Episode 6 is rejected at the
generation boundary. Agent Error and Transient Failure therefore produce no
permanent candidate.

The four patch families change agent-visible behavior without changing the
runtime: ICD transforms requests, RSD maps response fields, WPD declares and
executes prerequisite workflows, and SED declares read-after-write or
new-resource confirmation policies. RFC 6902-style operations are atomically
applied to an isolated `SpecOverlay`; semantic behavior is carried in explicit
`x-driftguard-*` extensions. Accepted patches are keyed in `PatchRegistry` by
tool, source fingerprint, localized path, and version—never by a benchmark
family or drift ID.

Every candidate passes the lifecycle
`PROPOSED → STATIC_VALIDATED → REPAIR_VALIDATED → REGRESSION_VALIDATED
→ SAFETY_VALIDATED → ACCEPTED`. Static validation checks fingerprint,
JSON Pointer scope, OpenAPI 3.1 validity, operation IDs, and security. Immediate
repair replays from a clean pre-failure state against the real Persistent Drift
runtime and uses the task evaluator. Three independent regression forks,
side-effect and tool-budget safety checks, and semantic minimality checks must
all pass. Any failure rejects the patch and rolls back the overlay.

Only after acceptance does the runner execute the held-out Episode 6 twice
from identical initial state: once without the patch and once with the accepted
patch. Future evidence cannot rank, validate, accept, or rewrite a candidate.
The ground-truth evaluator is constructed last and performs semantic rather
than textual patch comparison. The report also reruns Oracle, Phase 6, and
Phase 7 checks and verifies protected-file and canonical-handler hashes.

Run:

```bash
python scripts/run_healing_conformance.py
python scripts/run_healing_conformance.py --family M01
python scripts/run_healing_conformance.py --output results/healing/custom.json
python scripts/run_attribution_conformance.py
python scripts/run_injection_conformance.py
python scripts/run_oracle.py
pytest -q
```

The formal result is written to
`results/healing/healing_conformance_v1.json` and reports proposal prevention,
semantic correctness, immediate repair, future-task transfer, regression,
safety, minimality, deterministic replay, and per-category results. Phase 8
uses no LLM and does not claim to have completed the later LLM-agent experiment.

## Phase 9: LLM Tool-Agent Experiment Harness

Phase 9 adds a provider-neutral tool agent and a reproducible experiment
harness. `MockProvider` is the default and performs no network access. An
OpenAI-compatible endpoint can be configured through
`DRIFTGUARD_LLM_BASE_URL`, `DRIFTGUARD_LLM_MODEL`, and
`DRIFTGUARD_LLM_API_KEY`; the key is read only at runtime and is excluded from
prompts, cache keys, manifests, checkpoints, and records. Real-provider runs
also require the explicit `--allow-real-api` flag.

The common agent loop receives a natural-language task and displayed OpenAPI,
emits a schema-validated `TOOL_CALL`, `FINAL_ANSWER`, `REQUEST_PROBE`, or
`ABSTAIN` action, validates every tool and argument locally, executes through
the real sandbox/injection chain, and records visible observations. Only the
deterministic `TaskEvaluator` can mark a task successful; a model's statement
of success is not sufficient.

Recovery methods share the same base prompt, model, state, task, injection, and
budgets:

- `standard`: ordinary bounded continuation without a recovery framework.
- `retry_only`: one semantically exact tool retry.
- `reflection`: current-failure reflection with task-local memory only.
- `validation_guided`: displayed validation plus the current visible response,
  without persistent repair.
- `driftguard_llm`: LLM attribution/proposal with deterministic eligibility and
  Phase 8 validation; rejected LLM output is never replaced by a symbolic
  answer.
- `driftguard_symbolic`: the existing deterministic DriftGuard system version.
- `oracle_symbolic_upper_bound`: a separately labeled upper bound, never a real
  LLM result.

Every method uses the same `ExperimentBudget` for LLM calls, tool calls, probes,
total interactions, tokens, format repairs, wall time, and output size. Provider
HTTP retries remain separate from tool-level retries. Cache keys include model
configuration, prompt/schema/message hashes, public scenario, episode, method,
and repetition, but never credentials. Records are written atomically and
`--resume` skips completed records. The immutable manifest records the Git
commit and dirty state, benchmark and prompt hashes, model capabilities,
budgets, dependencies, selection, seed, and cache policy.

Component mode evaluates attribution/localization/patch interfaces from fixed
Agent-visible evidence. End-to-end mode begins with task planning and real tool
execution. Their metrics are stored in separate method/mode groups.

Run the offline smoke test and validation with:

```bash
python scripts/run_llm_experiment.py --config configs/experiments/phase9_mock_smoke.yaml
python scripts/run_llm_experiment.py --config configs/experiments/phase9_mock_smoke.yaml --mode component
python scripts/run_llm_experiment.py --config configs/experiments/phase9_mock_smoke.yaml --mode end_to_end
python scripts/run_llm_experiment.py --config configs/experiments/phase9_mock_smoke.yaml --resume
python scripts/validate_experiment_results.py results/experiments/phase9_mock_smoke
```

Raw provider/cache material is stored only under ignored experiment cache
directories. Public records exclude evaluator metadata, hidden contracts,
initial state, credentials, variant codes, expected patches, and chain of
thought. The Phase 9 committed smoke report is generated exclusively by
`MockProvider` and is explicitly labeled:

`Real LLM experiment status: NOT RUN`

No paid or formal multi-model experiment is performed in Phase 9.

## Phase 10A: Cost-Controlled Real-Model Pilot

Phase 10A fixes the real providers to `deepseek-v4-flash` at
`https://api.deepseek.com` and `qwen3.7-plus` at the DashScope OpenAI-compatible
endpoint. Credentials are loaded from the ignored project-root `.env` through
the independent `DEEPSEEK_API_KEY` and `DASHSCOPE_API_KEY` variables. The CLI
credential preflight prints only `configured` or `missing`; keys are excluded
from prompts, cache keys, errors, manifests, records, and raw artifacts.

Provider adapters explicitly disable thinking with the vendor-specific
parameter, use strict JSON fallback when JSON Schema mode is unavailable, and
preflight text, JSON, tool calling, usage reporting, and the actual returned
model ID. A versioned conservative pricing snapshot reserves worst-case cost
before each HTTP attempt. The Pilot soft/hard limits are CNY 35/CNY 50. Main
and ablation configurations have CNY 800/CNY 1000 controls but cannot be run by
the Pilot entrypoint and require separate authorization.

Run the credential and API capability preflight, then the authorized Pilot:

```bash
python scripts/run_llm_experiment.py \
  --config configs/experiments/phase10_real_pilot.yaml \
  --preflight --allow-real-api

python scripts/run_llm_experiment.py \
  --config configs/experiments/phase10_real_pilot.yaml \
  --allow-real-api

python scripts/run_llm_experiment.py \
  --config configs/experiments/phase10_real_pilot.yaml \
  --allow-real-api --resume

python scripts/validate_experiment_results.py results/experiments/phase10/pilot
python scripts/analyze_phase10.py --input results/experiments/phase10/pilot --stage pilot
```

Prompt-format repairs are append-only experiment attempts. The original
component failure remains under `pilot`, the schema-complete component prompt
attempt remains under `pilot_v2`, and the end-to-end-only AgentAction prompt
attempt uses `phase10_real_pilot_v3.yaml`, `pilot_v3`, and its own cache
namespace. The v3 attempt never reruns the successful standalone component
records. Its gated sequence is DeepSeek five-method canary, DeepSeek 60-record
end-to-end stage, Qwen five-method canary, then Qwen 60-record end-to-end
stage:

```bash
python scripts/run_llm_experiment.py \
  --config configs/experiments/phase10_real_pilot_v3.yaml \
  --allow-real-api --end-to-end-canary deepseek

python scripts/run_llm_experiment.py \
  --config configs/experiments/phase10_real_pilot_v3.yaml \
  --allow-real-api --resume
```

All attempts share the same cumulative Pilot cost ledger and CNY 50 hard
limit. A canary requires five schema-valid records and at least one parsed and
executed `TOOL_CALL`; an infrastructure error rate above 20% halts subsequent
provider stages.

Pilot records are always labeled `PILOT` and `DEVELOPMENT_ONLY`; component,
end-to-end, symbolic upper-bound, cache, preflight, cost, and analysis outputs
remain separate. The main all-60, held-out-48, and ablation YAML files are
configuration artifacts only and are not executed in Phase 10A.

`Full real-model experiment status: NOT RUN`

## Phase 11: Frozen SpecDriftBench Held-out Component Protocol

The Development Canary remains a separate 36-record experiment over families
`M01`, `M06`, `M11`, and `M16`. The frozen held-out Component experiment uses
the other 16 families (`M02`–`M05`, `M07`–`M10`, `M12`–`M15`, and
`M17`–`M20`) and is exactly:

```text
16 held-out families × 3 variants × 3 Evidence Views × 3 Providers × 1 repetition
= 432 records
```

DeepSeek, Qwen, and Kimi each receive 144 records. Each of the 144 groups keyed
by Provider, family, and variant contains the same public scenario under
`FIRST_FAILURE`, `RETRY_HISTORY`, and `FULL_EVIDENCE`. Evidence is monotone
within a group and ground truth remains evaluator-only. This is one execution,
not 144 records repeated three times.

The versioned configuration defaults to `run_authorized: false`. A future real
run is fail-closed unless an authorized copy of that configuration, the
`--allow-real-api` flag, and the exact `RUN-EXACTLY-432` confirmation are all
present. It also requires a clean Git worktree. The per-attempt soft and hard
limits are CNY 15 and CNY 20, with a CNY 50 global hard limit and atomic Ledger
reservation/settlement.

Validate the frozen plan or execute the disposable offline FakeProvider path:

```bash
python scripts/run_specdriftbench_component_heldout.py --validate-config
python scripts/run_specdriftbench_component_heldout.py \
  --fake-validate --output /path/outside/formal/results
```

Formal attempts use isolated cache, checkpoint, result, and Manifest
namespaces. Resume verifies their identity and never reruns completed records;
replay requires every original cache entry, makes no Provider calls, and
fails closed on a cache miss or semantic mismatch. The analysis endpoints,
denominators, paired comparisons, and family-clustered bootstrap (10,000
resamples, seed 20260718, 95% confidence) are frozen in
`benchmark/analysis/specdriftbench_heldout432_analysis_plan_v1.json` before any
real result exists.

`Held-out 432 real API status: NOT RUN`
