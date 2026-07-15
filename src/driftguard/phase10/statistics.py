from __future__ import annotations

import math
import random
from typing import Any, Callable


def mcnemar_exact(first: list[bool], second: list[bool]) -> dict[str, float | int]:
    if len(first) != len(second):
        raise ValueError("paired vectors must have the same length")
    b = sum(a and not c for a, c in zip(first, second))
    c = sum(not a and c for a, c in zip(first, second))
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return {"first_only": b, "second_only": c, "discordant": n, "p_value": p}


def paired_family_bootstrap(
    pairs: list[dict[str, Any]],
    metric: Callable[[dict[str, Any]], float],
    iterations: int = 10_000,
    seed: int = 20260715,
) -> dict[str, float]:
    if not pairs:
        return {"mean_difference": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    by_family: dict[str, list[float]] = {}
    for pair in pairs:
        by_family.setdefault(pair["public_family_id"], []).append(metric(pair))
    family_means = [sum(values) / len(values) for values in by_family.values()]
    observed = sum(family_means) / len(family_means)
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sample = [rng.choice(family_means) for _ in family_means]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    return {"mean_difference": observed, "ci_low": low, "ci_high": high}


def holm_correction(p_values: list[float]) -> list[float]:
    count = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * count
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[index] = running
    return adjusted
