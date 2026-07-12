from __future__ import annotations

from copy import deepcopy

import pytest

from driftguard.sandbox import SandboxService, StateStore


@pytest.fixture
def service():
    return SandboxService()


@pytest.fixture
def initial_state(service):
    return service.store.snapshot()


def service_from_state(state):
    return SandboxService(store=StateStore.from_state(deepcopy(state)))
