from __future__ import annotations

import hashlib
import json

from driftguard.llm import LLMCache, ProviderResponse
from driftguard.phase10.consistency_audit import source_snapshot
from driftguard.phase10.v4_canary import ReadOnlyCache, _policy_gate


def _tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def test_v4_replay_cache_reads_without_mutating(tmp_path):
    cache = LLMCache(tmp_path)
    response = ProviderResponse("r1", "fixed-model", {"ok": True}, json.dumps({"ok": True}), 11, 3, 14, 1.0, 1, "stop")
    cache.put("key", response)
    before = _tree_hash(tmp_path)
    loaded = ReadOnlyCache(tmp_path).get("key")
    assert loaded is not None and loaded.cached and loaded.raw_text == response.raw_text
    assert ReadOnlyCache(tmp_path).get("missing") is None
    assert _tree_hash(tmp_path) == before


def test_v4_policy_gates_require_the_real_method_specific_hooks():
    standard = _policy_gate("standard", {"_policy_events": [], "_actions": [], "_driftguard_path": {}})
    assert standard["passed"]
    retry = _policy_gate("retry_only", {
        "_policy_events": [{"exact_retry": True}],
        "_actions": [{"action_type": "EXACT_RETRY"}],
        "_driftguard_path": {},
    })
    assert retry["passed"]
    assert not _policy_gate("retry_only", {"_policy_events": [], "_actions": [], "_driftguard_path": {}})["passed"]
    reflection = _policy_gate("reflection", {
        "_policy_events": [{"context_types": ["reflection"]}], "_actions": [], "_driftguard_path": {},
    })
    assert reflection["passed"]
    guided = _policy_gate("validation_guided", {
        "_policy_events": [{"failure_kind": "local_validation", "context_types": ["local_validation"]}],
        "_actions": [], "_driftguard_path": {},
    })
    assert guided["passed"]
    driftguard = _policy_gate("driftguard_llm", {
        "_policy_events": [], "_actions": [],
        "_driftguard_path": {
            "evidence_entered": True, "attribution_entered": True,
            "eligibility_entered": True, "healing_entered": True,
        },
    })
    assert driftguard["passed"]


def test_clean_canary_source_snapshot_is_hash_addressed():
    snapshot = source_snapshot()
    assert len(snapshot["source_snapshot_hash"]) == 64
    assert "src/driftguard/agents/controller.py" in snapshot["files"]
