# Phase 10A pilot_v3 End-to-End Root-Cause Audit

This audit is cache-only and offline. It made zero provider calls and added zero cost.

## 1. Failure funnel

- Records: 120; Task Success: 0
- Terminations: `{"ABSTAINED": 50, "BUDGET_EXHAUSTED": 6, "INVALID_ARGUMENTS": 45, "SAFETY_BLOCKED": 19}`
- Action types: `{"ABSTAIN": 50, "REQUEST_PROBE": 37, "TOOL_CALL": 319}`
- Never sandboxed: 45; sandboxed: 75
- Local validation before Sandbox: 45
- Recovery hook never activated: 45

## 2. DeepSeek stopping and safety

`{"budget_exhausted": 2, "controller_probe_safety_block": 17, "format_valid_semantically_empty": 0, "internal_stop": 0, "model_abstain": 41, "model_final_answer": 0, "model_refusal": 0, "parameter_safety_rejection": 0, "write_tool_globally_forbidden": 0}`

No normal write was blocked. `SAFETY_BLOCKED` is the Controller rejecting REQUEST_PROBE for methods whose policy has `allows_probe=False`.
- `standard`: M01-AE `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/af86f9ecc1172974a3853d56.json` → ABSTAIN / ToolAgentController.run -> AgentAction.ABSTAIN -> break; M01-TF `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/672dfb2408bd945f534dcebd.json` → REQUEST_PROBE / ToolAgentController.run -> REQUEST_PROBE with policy.allows_probe=False; M06-AE `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/73cccbc4b81584c25f3c9ab0.json` → ABSTAIN / ToolAgentController.run -> AgentAction.ABSTAIN -> break
- `retry_only`: M01-AE `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/ae8cb05d92da47102f11bcee.json` → ABSTAIN / ToolAgentController.run -> AgentAction.ABSTAIN -> break; M06-AE `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/6f5c534b5a3b18c91b396a28.json` → ABSTAIN / ToolAgentController.run -> AgentAction.ABSTAIN -> break; M11-AE `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/036fb72c8a4a4f09b6556578.json` → ABSTAIN / ToolAgentController.run -> AgentAction.ABSTAIN -> break
- `reflection`: M01-AE `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/c5fcbd51977f56f444055e9d.json` → ABSTAIN / ToolAgentController.run -> AgentAction.ABSTAIN -> break; M06-AE `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/312471cc1ec75e8275c17cd5.json` → REQUEST_PROBE / ToolAgentController.run -> REQUEST_PROBE with policy.allows_probe=False; M11-AE `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/19fb05132ee18b41a4560f11.json` → ABSTAIN / ToolAgentController.run -> AgentAction.ABSTAIN -> break
- `validation_guided`: M01-AE `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/58f291ffd9c9bbc606c94616.json` → ABSTAIN / ToolAgentController.run -> AgentAction.ABSTAIN -> break; M06-AE `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/2d9f4b24325ad4c212a8fb7b.json` → REQUEST_PROBE / ToolAgentController.run -> REQUEST_PROBE with policy.allows_probe=False; M06-PD `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/0fde90aa7c6ff2f14d7ab17a.json` → ABSTAIN / ToolAgentController.run -> AgentAction.ABSTAIN -> break
- `driftguard_llm`: M01-AE `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/61ac74f42edc0a3225053ef7.json` → ABSTAIN / ToolAgentController.run -> AgentAction.ABSTAIN -> break; M06-AE `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/d37c8e7a123b67b5127fabd2.json` → ABSTAIN / ToolAgentController.run -> AgentAction.ABSTAIN -> break; M11-AE `results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/ce82746cccc4d7ceb0145284.json` → ABSTAIN / ToolAgentController.run -> AgentAction.ABSTAIN -> break

## 3. Qwen parameter errors

- Local validation failures: 45
- Categories: `{"ID does not exist": 48, "invalid enum": 0, "missing required field": 15, "nested/flat body mismatch": 0, "permission actor error": 0, "unknown field": 0, "unresolved placeholder": 0, "wrong field name": 0, "wrong tool": 14, "wrong type": 30}`
- Top fields: `[{"count": 30, "field": "issue_id"}, {"count": 15, "field": "title"}]`
- `standard`: M06-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/7c310c51e74457bf984ef787.json` → create_issue / 'title' is a required property; M11-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/0652a9e6ebe149bad2789ba2.json` → get_issue / '101' is not of type 'integer'; M16-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/f279a347f785463033c1e69c.json` → close_issue / '102' is not of type 'integer'
- `retry_only`: M06-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/22fa316ed95509c8bb45ff99.json` → create_issue / 'title' is a required property; M11-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/2c39c7e4e030067ccb52c838.json` → get_issue / '101' is not of type 'integer'; M16-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/0b82171ed516d536e4605355.json` → close_issue / '102' is not of type 'integer'
- `reflection`: M06-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/7f5052c8d905543417f281b2.json` → create_issue / 'title' is a required property; M11-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/afe641accd279c32371e17f9.json` → get_issue / '101' is not of type 'integer'; M16-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/bd6a7fd114fd9180790c00a1.json` → close_issue / '102' is not of type 'integer'
- `validation_guided`: M06-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/8211d2dd3a26437ba92e5f2e.json` → create_issue / 'title' is a required property; M11-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/bbda7d83c5eecf2ce44d92e1.json` → get_issue / '101' is not of type 'integer'; M16-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/1543bc1938d1e9d075e59b8c.json` → close_issue / '102' is not of type 'integer'
- `driftguard_llm`: M06-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/764d13e50f33498f4a882a0f.json` → create_issue / 'title' is a required property; M11-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/c49b4cb2057bb4686df2d0b9.json` → get_issue / '101' is not of type 'integer'; M16-AE `results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/11571917b17b9fa4a1481993.json` → close_issue / '102' is not of type 'integer'

## 4. Task binding

- Concrete text: 12/12
- S0 visible: 0/12
- Actor visible: 0/12
- Natural-language goal aligned with evaluator: 3/12

## 5. Tool catalog versus Registry

- Tool IDs match: True
- Tools with complete structural diff=0: 0/12
- The model receives only tool_id, summary, path and method; input and response schemas are omitted.
- `get_repository` missing: description, required_fields, optional_fields, types, enums, defaults, request_body_hierarchy, response_structure, permission, refs_resolved, path_and_body_merged
- `update_repository` missing: description, required_fields, optional_fields, types, enums, defaults, request_body_hierarchy, response_structure, permission, refs_resolved, path_and_body_merged
- `create_issue` missing: description, required_fields, optional_fields, types, enums, defaults, request_body_hierarchy, response_structure, permission, refs_resolved, path_and_body_merged
- `get_issue` missing: description, required_fields, optional_fields, types, enums, defaults, request_body_hierarchy, response_structure, permission, refs_resolved, path_and_body_merged
- `assign_issue` missing: description, required_fields, optional_fields, types, enums, defaults, request_body_hierarchy, response_structure, permission, refs_resolved, path_and_body_merged
- `close_issue` missing: description, required_fields, optional_fields, types, enums, defaults, request_body_hierarchy, response_structure, permission, refs_resolved, path_and_body_merged
- `trigger_pipeline` missing: description, required_fields, optional_fields, types, enums, defaults, request_body_hierarchy, response_structure, permission, refs_resolved, path_and_body_merged
- `get_pipeline_status` missing: description, required_fields, optional_fields, types, enums, defaults, request_body_hierarchy, response_structure, permission, refs_resolved, path_and_body_merged
- `retry_pipeline` missing: description, required_fields, optional_fields, types, enums, defaults, request_body_hierarchy, response_structure, permission, refs_resolved, path_and_body_merged
- `add_member` missing: description, required_fields, optional_fields, types, enums, defaults, request_body_hierarchy, response_structure, permission, refs_resolved, path_and_body_merged
- `get_member` missing: description, required_fields, optional_fields, types, enums, defaults, request_body_hierarchy, response_structure, permission, refs_resolved, path_and_body_merged
- `update_member_role` missing: description, required_fields, optional_fields, types, enums, defaults, request_body_hierarchy, response_structure, permission, refs_resolved, path_and_body_merged

## 6. Controller and recovery

Legal calls execute and Sandbox failures become observations. Local validator failures terminate immediately, are never shown on another model turn, and bypass every recovery policy.

## 7. Policy hook counts

| Method | Hook calls | Records stopped before hook | Exact retry decisions | Validation-error contexts |
|---|---:|---:|---:|---:|
| standard | 59 | 9 | 0 | 0 |
| retry_only | 69 | 9 | 15 | 0 |
| reflection | 52 | 9 | 0 | 0 |
| validation_guided | 57 | 9 | 0 | 0 |
| driftguard_llm | 52 | 9 | 0 | 0 |

## 8. TaskEvaluator recalculation

- Exact recalculation matches: 120/120
- No observed model success was incorrectly returned false; however, the smoke evaluator is too weak and can return false positives for multi-step tasks.
- Reason codes: `{"ANSWER_ASSERTION_FAILED": 120, "TOOL_BUDGET_EXCEEDED": 12, "TOOL_EXECUTION_FAILED": 75}`

## 9. Offline Oracle downstream control

- TaskEvaluator pass: 12/12
- Natural-language business goal complete: 3/12
- M01-AE: evaluator=True, business_goal=True, sandbox_calls=1
- M01-TF: evaluator=True, business_goal=True, sandbox_calls=1
- M01-PD: evaluator=True, business_goal=True, sandbox_calls=1
- M06-AE: evaluator=True, business_goal=False, sandbox_calls=1
- M06-TF: evaluator=True, business_goal=False, sandbox_calls=1
- M06-PD: evaluator=True, business_goal=False, sandbox_calls=1
- M11-AE: evaluator=True, business_goal=False, sandbox_calls=2
- M11-TF: evaluator=True, business_goal=False, sandbox_calls=2
- M11-PD: evaluator=True, business_goal=False, sandbox_calls=2
- M16-AE: evaluator=True, business_goal=False, sandbox_calls=1
- M16-TF: evaluator=True, business_goal=False, sandbox_calls=1
- M16-PD: evaluator=True, business_goal=False, sandbox_calls=1

## 10. Recorded-action counterfactual

- Qwen invalid records checked: 45
- Minimal structural fixes locally valid: 30
- Sandbox-executable after structural fix: 30
- Median edits for fixable records: 1.0
- DeepSeek actionable observations before stop: 60/60

## 11. Ranked root causes

### 1. P1 — Prompt/tool catalog rendering

AgentContextBuilder replaces the displayed OpenAPI with a four-field tool summary. Required arguments, types, enums, defaults, response schemas and permissions are absent for all 12 tools.

Minimal fix: Render the resolved Agent-visible input/response contract and actor capability for every tool; keep the existing strict AgentAction envelope.

Evidence: `["src/driftguard/agents/context_builder.py::_tool_summary", "tool_catalog_audit.tools_with_missing_schema_information=12", "task_instantiation.requires_undiscoverable_repo_id_scenarios=12"]`

Representative records: `["results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/af86f9ecc1172974a3853d56.json", "results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/ae8cb05d92da47102f11bcee.json", "results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/c5fcbd51977f56f444055e9d.json"]`

Behavior-affecting: `true`

### 2. P2 — Controller feedback loop

A local InputValidationError is written to TaskMemory and then the controller immediately breaks, so the model never receives the concrete validator error and no recovery policy hook can run.

Minimal fix: Within the unchanged interaction budget, append the validator error and continue to one next model action; route the failure through the selected policy hook.

Evidence: `["src/driftguard/agents/controller.py local InputValidationError branch", "qwen_parameter_analysis.local_validation_failures=45", "controller_loop.validation_error_returned_to_model=0"]`

Representative records: `["results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/7c310c51e74457bf984ef787.json", "results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/22fa316ed95509c8bb45ff99.json", "results/experiments/phase10/pilot_v3/end_to_end/dashscope-qwen3.7-plus/records/7f5052c8d905543417f281b2.json"]`

Behavior-affecting: `true`

### 3. P0 — Task instantiation/evaluator contract

The smoke task reduces all natural-language goals to tool_succeeded=true with no state assertions. M06/M11/M16 are multi-step or state-sensitive, and M16 names issue 102 while the SED-01 benchmark arguments target issue 101.

Minimal fix: Instantiate task-specific success assertions and consistent bound entity IDs before any new Pilot; validate them with the 12-task offline Oracle control.

Evidence: `["src/driftguard/experiments/scenario_runner.py::_smoke_task", "task_instantiation.task_evaluator_goal_alignment_scenarios=3/12", "oracle_downstream_control business-goal count versus evaluator pass count"]`

Representative records: `["results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/73cccbc4b81584c25f3c9ab0.json", "results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/6f5c534b5a3b18c91b396a28.json", "results/experiments/phase10/pilot_v3/end_to_end/deepseek-deepseek-v4-flash/records/312471cc1ec75e8275c17cd5.json"]`

Behavior-affecting: `true`

## 12. Recommendation

Do not authorize Phase 10B and do not create Pilot v4 yet. First implement and pass the offline P0/P1/P2 gates. If later authorized, use a 10-record canary (five methods x two providers).

Estimated new canary cost after authorization: CNY 0.13–0.40 (estimate only). Current audit cost: CNY 0.

**Full real-model experiment status: NOT RUN**
