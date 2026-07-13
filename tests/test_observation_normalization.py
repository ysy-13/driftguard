from conftest import injection_context
from driftguard.injection import ObservationNormalizer
from driftguard.runners.injection_conformance_runner import CASE_ARGUMENTS
from driftguard.sandbox import SandboxService


def test_normalized_observation_matches_family_signature(injection_catalog):
    drift_id = "RSD-04"
    context = injection_context(injection_catalog, drift_id, "PD")
    service = SandboxService(execution_context=context)
    before = service.store.snapshot()
    result = service.call_tool("get_repository", CASE_ARGUMENTS[drift_id], "agent_admin")
    signature = injection_catalog[1][drift_id]["shared_observation_signature"]
    observation = ObservationNormalizer().normalize(
        signature, result, before, service.store.snapshot(), CASE_ARGUMENTS[drift_id]
    )
    assert observation == signature
    forbidden = {"variant", "source_drift_id", "ground_truth_label", "expected_patch"}
    assert forbidden.isdisjoint(observation)
