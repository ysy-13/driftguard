import pytest

from driftguard.contracts.loader import PROJECT_ROOT
from driftguard.llm import ProviderRequest, ProviderResponse
from driftguard.phase10.config import Phase10Config
from driftguard.phase10.costs import CostBudgetManager, CostHardLimit
from driftguard.phase10.pricing import PricingCatalog
from driftguard.phase10.statistics import holm_correction, mcnemar_exact, paired_family_bootstrap


CONFIG = Phase10Config.load(PROJECT_ROOT / "configs/experiments/phase10_real_pilot.yaml")
REQUEST = ProviderRequest(({"role": "user", "content": "x"},), {}, "0" * 64, "public-0123456789ab", 1, "standard", 0)


def test_deepseek_usd_conversion_and_qwen_cny_pricing():
    catalog = PricingCatalog.load_default()
    deepseek, qwen = CONFIG.models
    assert catalog.estimate_cny(deepseek, 1_000_000, 1_000_000) == pytest.approx((0.14 + 0.28) * 7.3)
    assert catalog.estimate_cny(qwen, 1_000_000, 1_000_000) == pytest.approx(10.0)


def test_worst_case_is_reserved_before_call_and_reconciled_to_usage():
    manager = CostBudgetManager(PricingCatalog.load_default(), 35, 50, 60_000, 12_000)
    reservation = manager.reserve(CONFIG.models[1], REQUEST)
    assert manager.reserved_cny > 0 and manager.api_attempts == 1 and manager.spent_cny == 0
    response = ProviderResponse("r", CONFIG.models[1].model_id, None, "{}", 100, 20, 120, 1, 1, "stop")
    manager.settle(reservation, response)
    assert manager.reserved_cny == pytest.approx(0) and manager.spent_cny > 0


def test_soft_warning_and_hard_stop_are_enforced_before_request():
    catalog = PricingCatalog.load_default()
    manager = CostBudgetManager(catalog, 0.01, 0.2, 60_000, 12_000)
    reservation = manager.reserve(CONFIG.models[1], REQUEST)
    assert manager.soft_warning
    manager.release(reservation)
    blocked = CostBudgetManager(catalog, 0.01, 0.05, 60_000, 12_000)
    with pytest.raises(CostHardLimit):
        blocked.reserve(CONFIG.models[1], REQUEST)
    assert blocked.api_attempts == 0


def test_mcnemar_exact_known_discordant_pairs():
    result = mcnemar_exact([True, True, False, False], [False, True, True, True])
    assert result["first_only"] == 1 and result["second_only"] == 2
    assert 0 <= result["p_value"] <= 1


def test_family_paired_bootstrap_is_deterministic():
    pairs = [
        {"public_family_id": "f1", "difference": 1.0},
        {"public_family_id": "f1", "difference": 0.0},
        {"public_family_id": "f2", "difference": -1.0},
    ]
    first = paired_family_bootstrap(pairs, lambda row: row["difference"], 1000, 7)
    second = paired_family_bootstrap(pairs, lambda row: row["difference"], 1000, 7)
    assert first == second and first["ci_low"] <= first["mean_difference"] <= first["ci_high"]


def test_holm_correction_preserves_input_order_and_monotonicity():
    adjusted = holm_correction([0.04, 0.01, 0.03])
    assert len(adjusted) == 3 and all(0 <= item <= 1 for item in adjusted)
    assert adjusted[1] <= adjusted[2] <= adjusted[0]
