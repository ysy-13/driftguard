from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from driftguard.evidence.collector import stable_hash
from driftguard.sandbox import SandboxService, StateStore
from .probes import ProbePlan, SAFE_PROBE_TYPES, UnsafeProbeError


class ProbeExecutor:
    def __init__(self):
        self.unsafe_probe_count = 0

    def execute(
        self,
        plan: ProbePlan,
        main_service: SandboxService,
        operation: Callable[[SandboxService], Any],
    ) -> dict[str, Any]:
        if plan.probe_type not in SAFE_PROBE_TYPES:
            self.unsafe_probe_count += 1
            raise UnsafeProbeError(f"unsafe probe type: {plan.probe_type}")
        main_before = stable_hash(main_service.store.snapshot())
        fork = SandboxService(store=StateStore.from_state(deepcopy(main_service.store.snapshot())))
        value = operation(fork)
        main_after = stable_hash(main_service.store.snapshot())
        if main_before != main_after:
            raise RuntimeError("probe polluted main business state")
        return {
            "probe_type": plan.probe_type,
            "main_state_hash_before": main_before,
            "main_state_hash_after": main_after,
            "state_unchanged": True,
            "result": value,
        }
