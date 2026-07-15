from __future__ import annotations

from dataclasses import dataclass
import threading
import uuid
from typing import Any

from driftguard.llm import ModelConfig, ProviderRequest, ProviderResponse

from .pricing import PricingCatalog


class CostHardLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    amount_cny: float
    config: ModelConfig
    request: ProviderRequest


class CostBudgetManager:
    def __init__(
        self,
        catalog: PricingCatalog,
        soft_limit_cny: float,
        hard_limit_cny: float,
        max_input_tokens_per_call: int,
        max_output_tokens_per_call: int,
        initial_spent_cny: float = 0.0,
        initial_api_attempts: int = 0,
        initial_input_tokens: int = 0,
        initial_output_tokens: int = 0,
    ):
        self.catalog = catalog
        self.soft_limit_cny, self.hard_limit_cny = soft_limit_cny, hard_limit_cny
        self.max_input_tokens_per_call = max_input_tokens_per_call
        self.max_output_tokens_per_call = max_output_tokens_per_call
        self.spent_cny = float(initial_spent_cny)
        self.reserved_cny = 0.0
        self.soft_warning = False
        self.api_attempts = int(initial_api_attempts)
        self.initial_input_tokens = int(initial_input_tokens)
        self.initial_output_tokens = int(initial_output_tokens)
        self.events: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def reserve(self, config: ModelConfig, request: ProviderRequest) -> Reservation:
        amount = self.catalog.estimate_cny(
            config, self.max_input_tokens_per_call,
            min(config.max_output_tokens, self.max_output_tokens_per_call),
        )
        with self._lock:
            if self.spent_cny + self.reserved_cny + amount > self.hard_limit_cny:
                raise CostHardLimit("pilot hard cost limit would be exceeded before request")
            reservation = Reservation(uuid.uuid4().hex, amount, config, request)
            self.reserved_cny += amount
            self.api_attempts += 1
            if self.spent_cny + self.reserved_cny >= self.soft_limit_cny:
                self.soft_warning = True
            return reservation

    def settle(self, reservation: Reservation, response: ProviderResponse) -> None:
        actual = self.catalog.estimate_cny(
            reservation.config, response.input_tokens, response.output_tokens,
        )
        with self._lock:
            self.reserved_cny -= reservation.amount_cny
            self.spent_cny += actual
            if self.spent_cny >= self.soft_limit_cny:
                self.soft_warning = True
            self.events.append({
                "provider": reservation.config.provider,
                "configured_model": reservation.config.model_id,
                "actual_model": response.model,
                "public_scenario_id": reservation.request.public_scenario_id,
                "method": reservation.request.method,
                "mode": reservation.request.mode,
                "repetition": reservation.request.repetition,
                "seed": reservation.request.seed,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "estimated_cost_cny": actual,
                "provider_attempts": response.provider_attempts,
            })

    def release(self, reservation: Reservation) -> None:
        with self._lock:
            self.reserved_cny -= reservation.amount_cny

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "spent_cny": self.spent_cny,
                "reserved_cny": self.reserved_cny,
                "soft_limit_cny": self.soft_limit_cny,
                "hard_limit_cny": self.hard_limit_cny,
                "soft_warning": self.soft_warning,
                "api_attempts": self.api_attempts,
                "provider_reported_input_tokens": self.initial_input_tokens + sum(item["input_tokens"] for item in self.events),
                "provider_reported_output_tokens": self.initial_output_tokens + sum(item["output_tokens"] for item in self.events),
            }
