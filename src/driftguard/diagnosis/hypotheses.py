from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureHypothesis:
    name: str
    supporting_evidence_ids: tuple[str, ...]
    contradicted_by: tuple[str, ...] = ()


DEFAULT_HYPOTHESES = ("AE", "TF", "PD")
