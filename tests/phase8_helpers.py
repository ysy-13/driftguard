from __future__ import annotations

from functools import lru_cache

from driftguard.runners.attribution_conformance_runner import AttributionConformanceRunner


@lru_cache(maxsize=64)
def artifacts(family_id: str, variant: str = "PD"):
    runner = AttributionConformanceRunner()
    family = next(item for item in runner.matched["families"] if item["matched_case_id"] == family_id)
    scenario = next(item for item in family["scenarios"] if item["variant_code"] == variant)
    return runner._run_scenario(family, scenario, return_agent_artifacts=True)


def candidate(family_id: str = "M01"):
    from driftguard.healing import PatchCandidateGenerator

    value = artifacts(family_id)
    return PatchCandidateGenerator().generate(value["agent_view"], value["diagnosis"]).candidates[0]

