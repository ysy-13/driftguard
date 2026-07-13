from conftest import injection_context


def test_episode_scheduling_and_candidate_patch(injection_catalog):
    ae = injection_context(injection_catalog, "ICD-01", "AE", 2)
    assert not ae.agent_fault_active
    assert ae.injection_active("create_issue")
    ae.set_episode(3)
    assert ae.agent_fault_active
    ae.set_episode(4)
    assert not ae.agent_fault_active
    assert ae.injection_active("create_issue")
    pd = injection_context(injection_catalog, "ICD-01", "PD", 4)
    assert pd.injection_active("create_issue")


def test_transient_is_consumed_once(injection_catalog):
    context = injection_context(injection_catalog, "ICD-01", "TF")
    assert context.injection_active("create_issue")
    context.consume()
    assert not context.injection_active("create_issue")
