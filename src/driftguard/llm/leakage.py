from __future__ import annotations

import json
import re

from driftguard.evidence.leakage_guard import assert_agent_visible
from .models import ProviderRequest


FORBIDDEN_MESSAGE_TERMS = (
    "ground_truth_label", "expected_patch_ref", "source_drift_id", "variant_code",
    "runtimeprofile", "runtime_profile", "hidden runtime contract", "runtime_contract",
)
LEGAL_PREDICTION_TERMS = (
    "AGENT_ERROR", "TRANSIENT_FAILURE", "PERSISTENT_DRIFT",
    "ICD", "RSD", "WPD", "SED", "NONE",
)


def assert_provider_request_visible(request: ProviderRequest) -> None:
    if type(request) is not ProviderRequest:
        raise TypeError("provider boundary accepts only ProviderRequest")
    messages = [dict(item) for item in request.messages]
    # Output schemas legitimately enumerate the prediction vocabulary. Strip
    # only the explicitly delimited schema block before applying the ordinary
    # AgentView value-term guard; evaluator-only keys remain forbidden below.
    visible_messages = []
    for message in messages:
        copy = dict(message)
        copy["content"] = re.sub(
            r"<OUTPUT_SCHEMA>.*?</OUTPUT_SCHEMA>", "<OUTPUT_SCHEMA_REDACTED>",
            copy.get("content", ""), flags=re.DOTALL,
        )
        copy["content"] = re.sub(
            r"\b(?:" + "|".join(LEGAL_PREDICTION_TERMS) + r")\b",
            "LEGAL_PREDICTION_LABEL", copy["content"], flags=re.IGNORECASE,
        )
        assert_agent_visible(copy)
        visible_messages.append(copy)
    encoded = json.dumps(messages, sort_keys=True).lower()
    if any(term in encoded for term in FORBIDDEN_MESSAGE_TERMS):
        raise ValueError("provider request contains evaluator-only information")
