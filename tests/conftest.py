from __future__ import annotations

from copy import deepcopy

import pytest

from driftguard.sandbox import SandboxService, StateStore
from driftguard.runners.injection_conformance_runner import InjectionConformanceRunner
from driftguard.runtime import ExecutionContext, ExecutionMode, ExecutionProfile


@pytest.fixture
def service():
    return SandboxService()


@pytest.fixture
def initial_state(service):
    return service.store.snapshot()


def service_from_state(state):
    return SandboxService(store=StateStore.from_state(deepcopy(state)))


@pytest.fixture(scope="session")
def injection_catalog():
    runner = InjectionConformanceRunner()
    families = {family["source_drift_id"]: family for family in runner.families}
    cases = {case["drift_id"]: case for case in runner.drifts}
    return cases, families


def injection_context(injection_catalog, drift_id, mode, episode=3):
    cases, families = injection_catalog
    profile = ExecutionProfile(
        mode=ExecutionMode(mode),
        drift_case=cases[drift_id],
        family=families[drift_id],
        change_point=cases[drift_id]["change_point"],
    )
    return ExecutionContext(profile, episode_index=episode)
