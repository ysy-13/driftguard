# Phase 10A Post-repair Consistency Audit

Status: completed. The original v4 Canary is a `MIXED_VERSION_DEVELOPMENT_ARTIFACT`; the independent v4b Clean Canary passed. Phase 10B and the full real-model experiment remain `NOT RUN`.

## 1. Mid-run behavior changes

| Change | Before | Final behavior | Evidence time | Affected original v4 records | Model input | Arguments/Sandbox | AE/TF/PD and Task Success | Classification |
|---|---|---|---|---|---|---|---|---|
| Runtime normalization | Displayed validation inserted canonical defaults before the Runtime hook, so Runtime did not see the model's original omission. | With an execution context, the validated but unnormalized proposed arguments enter Runtime; context-free Phase 5 remains unchanged. | False-negative archive at 2026-07-15 23:12:23 +08; repaired before the post-repair retry/reflection re-execution. | Pre: DeepSeek standard. Post: the other nine current records, although DeepSeek retry reused provider responses generated around the boundary. | Initial prompt unchanged; later observations can change. | Yes; restores the first TF rejection and prevents defaults masking Runtime behavior. | Behavior can change for AE/TF/PD and can change Task Success. | Behavior-affecting. |
| Required-field default | `add_required` retained an obsolete default in the mutated contract copy, allowing local validation to fill the newly required field. | A newly required field loses that obsolete default only in the mutated copy. Canonical InputValidator behavior is unchanged. | Archived false-negative 23:28:59; source mtime 23:31:56 +08. | Pre: DeepSeek standard, retry_only, reflection. Post: DeepSeek validation_guided, driftguard_llm and all five Qwen records. | Yes, because the displayed Tool Catalog for the affected AE case changes. | Yes; omission now raises local/runtime validation instead of being filled. | AE requirement is enforceable; TF display stays canonical; PD runtime requirement is not bypassed. Task Success can change. | Behavior-affecting. |
| Agent Fault scope | The hook could be considered on unrelated tools and could recur on later target calls. | It activates only for AE, on the protocol episode, on the target tool, and is consumed once per record. | Completed before the first successful provider cache response at 23:04:01 +08. | All ten current v4 records are after this repair. | Initial prompt unchanged; subsequent feedback can change. | Yes, for AE only. | AE receives one scoped fault; TF/PD never receive it. Task Success can change. | Behavior-affecting. |

Implementation points: `ToolAgentController.run`, `_mutate_contract`, and `AgentFaultInjector`. Exact generation-time source hashes were not embedded in v4. Their overwritten preimages cannot be reconstructed honestly; the audit records them as `NOT_EMBEDDED_UNRECOVERABLE_PREIMAGE` instead of substituting current hashes.

## 2. Original v4 source-version proof

All records share Prompt v3 template SHA-256 `bae688c173db4f561162d402acef5661256c6bab5a1d20a65e96868c2ec8267f`, Schema v3 SHA-256 `27807767537dbcf83620642edc443525da61f08dfde6935f4ea11aafd1e9dcf1`, and semantic config hash `d8ad44bd3bf3102ca2ece14b5981e085214141fc23c788096235f84c7dd2d1ff`. Effective prompt/catalog hashes remain method/scenario specific.

| Provider | Method | Public scenario | Record | Generated +08 | Reconstructed semantic snapshot |
|---|---|---|---|---|---|
| DeepSeek | standard | public-adeb8ef1688f | f090807786162e951a7475e2 | 2026-07-15 23:12:14 | S1 pre-normalization repair |
| DeepSeek | retry_only | public-ba31a2577791 | 819771df8a1be8c211f2d30b | 2026-07-15 23:28:29 | S2 post-normalization, pre-default repair |
| DeepSeek | reflection | public-adeb8ef1688f | a99ad9cea713e48b6b93e1a2 | 2026-07-15 23:28:45 | S2 post-normalization, pre-default repair |
| DeepSeek | validation_guided | public-78014d069b15 | b859c7bd1a61abe4b793d02e | 2026-07-15 23:34:31 | S3 post-default repair |
| DeepSeek | driftguard_llm | public-ef4f2cacef0c | 31ff8920072a3186eb4d07c1 | 2026-07-15 23:34:51 | S3 post-default repair |
| Qwen | standard | public-adeb8ef1688f | b967aeb619e5547ff012dd8f | 2026-07-15 23:35:12 | S3 post-default repair |
| Qwen | retry_only | public-ba31a2577791 | b8663c62c7c10debda9a3ded | 2026-07-15 23:35:22 | S3 post-default repair |
| Qwen | reflection | public-adeb8ef1688f | e1884ceb19eaba93ba95d63f | 2026-07-15 23:35:45 | S3 post-default repair |
| Qwen | validation_guided | public-78014d069b15 | a5907deeca31edcc25ab9bd9 | 2026-07-15 23:35:56 | S3 post-default repair |
| Qwen | driftguard_llm | public-ef4f2cacef0c | 66b94edea363a088e134b771 | 2026-07-15 23:36:28 | S3 post-default repair |

Conclusion: at least three semantic source snapshots. The original v4 result set is marked `MIXED_VERSION_DEVELOPMENT_ARTIFACT`; combined Task Success is prohibited and is not used for method comparison.

## 3. Runtime normalization safety

- Every v4b parsed action stores `raw_model_action`, `normalized_action`, `normalization_diff`, and a reason code.
- 34 parsed actions were audited; semantic changes: 0; non-empty action-normalization diffs: 0.
- `tool`/`parameters` aliases are rejected and enter format repair; they are never silently converted to `tool_id`/`arguments`.
- No tool selection, tool ID, business argument, entity ID, Oracle value, ground truth, or hidden runtime rule is inserted by action normalization.
- Visible Schema defaulting is recorded separately as `displayed_input_audit`; 19 legitimate displayed-Schema default events occurred. They are not counted as Provider/action normalization.
- Any future semantic action-normalization diff is a model-output failure, not model success.

## 4. Required-field default audit

All five required negative cases passed:

1. missing required field without a default is rejected;
2. a default existing only in hidden Runtime is not used by Agent-visible validation;
3. ground truth is not an InputValidator input;
4. the old PD displayed spec accepts the old shape, while Runtime rejects omission of its new required field, so PD is not bypassed;
5. an incorrect entity ID (`R404`) remains `R404` and is never overwritten by a default.

Defaults are sourced only from the exact contract passed to InputValidator. Phase 5 canonical semantics remain intact.

## 5. Agent Fault scope matrix

The offline matrix contains 30 cells: AE/TF/PD x five methods x two providers.

| Variant | Cells | Active at episode 3 | Active at episode 4 | Provider/method differences | Result |
|---|---:|---:|---:|---:|---|
| AE | 10 | 10 | 0 | 0 | pass |
| TF | 10 | 0 | 0 | 0 | pass |
| PD | 10 | 0 | 0 | 0 | pass |

Reset restores the configured AE scenario for a new independent run but does not contaminate TF/PD or another context. Fault scope never depends on model success or Canary outcomes and is consistent with Phase 6.

## 6. Canonical protection and regressions

All seven protected hashes match the Phase 6 baseline:

- OpenAPI: `be893ee40a748410d49befe8ca45f645b47e9a770cf2bb7e0b90db9c71472267`
- initial state: `7cc0f4af8eb45d6e1a4a03741c2b44484b80736cf2ba6bd1f5c8e2c7fb7db3c0`
- tasks: `851425f77f4ad94633e5eb10937b783a51e9aa9f54955b408565064f3a788936`
- drift cases: `dda8f8c1759075bb52b14431b9ee5fcb6fbe93c1cf239a474548a6127228a00e`
- task/drift coverage: `f83a5ce5da1bd9a179dfdd1abfbb747469b29644c71304490df0750bd527be3c`
- matched failures: `58c921b754976ee16f29676030bfbb57de4e0ee2f7be14a4af8353d057296f7e`
- attribution protocol: `3e0779032b4613b9d17868ae1f2d3fe0c582214cc80bdb860a5a88aae69d4106`

The canonical handler directory has zero Git diff. It contains the existing handler modules implementing all 12 registered operations; no handler was copied or rewritten.

| Regression | Result |
|---|---|
| Phase 6 injection conformance | 60/60 first failures; AE/TF/PD each 20/20; Oracle 32/32 |
| Phase 7 attribution | 60/60; macro-F1 1.0; deterministic replay 60/60 |
| Phase 8 healing | PD proposed/accepted/correct 20/20; immediate/transfer 20/20 |
| Phase 9 mock | 168/168 records; validation passed; real LLM status NOT RUN |
| Canonical Oracle | 32/32; forbidden side effects 0; deterministic 32/32 |
| Offline Oracle | 12/12; Provider calls 0 |
| Scenario provenance | 60/60; placeholders/mismatches/hidden requirements 0 |
| Tool Catalog | 12/12; input-schema diff 0; leakage 0 |
| pytest final | 550 passed |
| `git diff --check` | passed |

## 7. v4b Clean Canary

Config file SHA-256: `57eef0df197e3d02d65f17ad484da58371e94823af19ca683306d84702e6b6b0`; semantic config hash: `9cbda4773b132e6348b018e84a7028fec0c51deee81dc3acf2e01d78fb20fcc4`.

Frozen source snapshot: `786a9274b467464626d0cdde31262cfa4b3efdaea31c90b554e10a00ea261726`. All 10 records contain this hash, and the same snapshot was verified again after the run.

| Provider | Method | Scenario | Termination | Task success | Tools / LLM / probes | Input / output tokens | API attempts | CNY |
|---|---|---|---|---:|---:|---:|---:|---:|
| DeepSeek | standard | M06-AE | BUDGET_EXHAUSTED | 0 | 4 / 4 / 0 | 50,180 / 298 | 5 | 0.065243458 |
| DeepSeek | retry_only | M01-TF | TASK_SUCCESS | 1 | 3 / 2 / 0 | 24,629 / 152 | 2 | 0.025481526 |
| DeepSeek | reflection | M06-AE | BUDGET_EXHAUSTED | 0 | 4 / 4 / 0 | 52,240 / 291 | 5 | 0.068575178 |
| DeepSeek | validation_guided | M01-AE | TASK_SUCCESS | 1 | 2 / 3 / 0 | 37,700 / 263 | 3 | 0.039066972 |
| DeepSeek | driftguard_llm | M06-PD | BUDGET_EXHAUSTED | 0 | 4 / 5 / 0 | 61,083 / 540 | 6 | 0.077157934 |
| Qwen | standard | M06-AE | BUDGET_EXHAUSTED | 0 | 4 / 4 / 0 | 48,329 / 489 | 5 | 0.125946000 |
| Qwen | retry_only | M01-TF | TASK_SUCCESS | 1 | 3 / 2 / 0 | 23,758 / 154 | 2 | 0.048748000 |
| Qwen | reflection | M06-AE | BUDGET_EXHAUSTED | 0 | 4 / 4 / 0 | 50,147 / 432 | 5 | 0.130798000 |
| Qwen | validation_guided | M01-AE | TASK_SUCCESS | 1 | 2 / 3 / 0 | 36,353 / 314 | 3 | 0.075218000 |
| Qwen | driftguard_llm | M06-PD | BUDGET_EXHAUSTED | 0 | 4 / 5 / 0 | 59,510 / 621 | 6 | 0.150300000 |

Acceptance: AgentAction 10/10; Sandbox 10/10; complete catalogs 10/10; correct task binding 10/10; infrastructure errors 0; state-isolation errors 0; unauthorized safety blocks 0; API-key/ground-truth leakage 0. Both providers activated all five required policy paths and achieved 2/5 Task Success. Cache replay matched 10/10 with zero Provider fallback, zero new attempts, and zero new cost.

Ledger delta from 4.396109052 CNY: 42 API attempts, 522,383 input tokens, 4,061 output tokens, and 0.806535068 CNY. Final cumulative ledger: 831 attempts, 3,580,832 input tokens, 84,182 output tokens, 5.202644120 CNY. The 1 CNY per-round soft limit, 2 CNY per-round hard limit, and 50 CNY total hard limit were not reached.

## 8. DriftGuard core-path diagnosis

Both clean DriftGuard records classified PD, entered the reported evidence/attribution flags, made four tool calls, made no probe, proposed no patch, and terminated on interaction budget.

The causes are not only model capability:

1. `REQUEST_PROBE` was present in the capability-narrowed schema on every DriftGuard model turn, and Controller did return the capability on each next round. Both models nevertheless emitted four `TOOL_CALL` actions. The policy prompt permits bounded probes but does not require an active probe when repeated failure crosses a threshold.
2. The attribution call does not consume the Controller's live evidence trace. `ExperimentScenarioRunner` independently regenerates the Phase 7 conformance artifact. That artifact has 12 events across episodes 1–5, historical evidence, two independent failures, a discriminative probe, and symbolic eligibility `ELIGIBLE`. Consequently the diagnostic model receives history and two failures, but those are not the actual four-tool Controller trajectory and are not counted in the record's probe budget.
3. The end-to-end `driftguard_llm` branch only performs LLM attribution. It leaves `patch_proposed = False`, never constructs `LLMPatchProposal`, never calls `policy.gate`, never invokes `guarded_llm_patch`, and never runs deterministic candidate validation/healing. `eligibility_entered` and `healing_entered` are currently flags derived from `predicted is not None`, not proof that those gates executed.
4. The AgentAction schema has TOOL_CALL, FINAL_ANSWER, REQUEST_PROBE, and ABSTAIN only. It has no patch-proposal action. The attribution schema also returns diagnosis, not a patch. Therefore no model in this path can emit a patch under the current public interface.
5. Budget exhaustion explains why the live tool loop stopped, but it does not explain the absent patch: patch generation is not wired and would remain absent with more tool-loop budget.

This is a common DriftGuard execution-chain gap, not a Provider-specific defect. The clean baseline policies are still comparable for their declared hooks, but the current end-to-end DriftGuard result is not evidence of a completed healing method.

## 9. Focused DriftGuard Development Canary proposal (not run)

Maximum eight records: DeepSeek and Qwen x `M01-PD`, `M06-PD`, `M11-PD`, `M16-PD`, method `driftguard_llm` only. No heldout48 data.

Before authorization, the focused protocol should require one live, provenance-linked chain per record:

1. two independent persistent failures from the actual Controller/session history;
2. at least one parsed and executed `REQUEST_PROBE`, with probe evidence returned to the next model turn;
3. LLM attribution over that live trace, with evidence-reference validation;
4. a separate, explicit LLM patch-proposal schema and call;
5. deterministic eligibility, safety, regression, and minimality validation;
6. immediate repair on the current task;
7. transfer validation on the future task;
8. record-level linkage proving every gate used the same live evidence and frozen source snapshot.

This proposal requires a separately authorized offline implementation and audit before any real calls. It was not executed in this round.

## 10. Recommendation

Do not authorize the complete Development Pilot yet: the v4b infrastructure and baseline policy hooks are stable, but the LLM DriftGuard end-to-end healing chain is incomplete and its current evidence source is decoupled from the live Controller trace. First implement and test that chain offline, then run the focused eight-record PD development canary.

Do not authorize Phase 10B yet.

**Phase 10B status: NOT RUN**

**Full real-model experiment status: NOT RUN**
