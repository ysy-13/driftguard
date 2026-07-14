from __future__ import annotations

from .models import ProviderRequest, ProviderResponse
from .provider import LLMProvider


class ProviderCallExecutor:
    """Names the provider boundary so its attempts cannot be counted as tool retries."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def execute(self, request: ProviderRequest) -> ProviderResponse:
        return self.provider.complete(request)

