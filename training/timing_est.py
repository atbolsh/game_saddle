"""Warmup vs steady-state timing helpers for datagen / train estimators.

A blended ``N * mean`` over-weights ramp. Full-run estimates are
``warmup_s + (N - n_warmup) * steady_per_item``.
"""

from __future__ import annotations

from typing import Any


def estimate_full_run(
    warmup_s: float,
    n_warmup: int,
    steady_per_item: float | None,
    n_total: int,
) -> float | None:
    """Seconds for ``n_total`` items given a measured warmup prefix.

    ``steady_per_item`` is seconds per item after warmup (throughput:
    leftover wall / leftover count). None if there is no leftover to
    measure — caller should say so rather than invent a rate.
    """
    if n_total <= 0:
        return 0.0
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be >= 0, got {n_warmup}")
    if n_total <= n_warmup:
        # Workload never left warmup. Scale the observed warmup wall
        # only if we have at least one item; do not pretend we know
        # a steady rate.
        if n_warmup == 0:
            return None
        return warmup_s * (n_total / n_warmup)
    if steady_per_item is None:
        return None
    return warmup_s + (n_total - n_warmup) * steady_per_item


def from_completion_times(
    t_start: float,
    completion_times: list[float],
    n_warmup: int,
) -> dict[str, Any]:
    """Split a run by completion timestamps (``perf_counter``).

    ``completion_times[i]`` is when item ``i`` finished. Warmup is the
    first ``n_warmup`` completions (clamped to the run). Steady-state
    is leftover wall after the last warmup completion, divided by the
    leftover count.
    """
    n = len(completion_times)
    if n == 0:
        return {
            "n_items": 0,
            "n_warmup": 0,
            "warmup_s": 0.0,
            "steady_n": 0,
            "steady_s": 0.0,
            "steady_per_item": None,
            "blended_per_item": None,
        }
    n_warm = max(1, min(int(n_warmup), n))
    ordered = sorted(completion_times)
    warmup_s = ordered[n_warm - 1] - t_start
    if warmup_s < 0:
        raise ValueError(
            f"warmup wall {warmup_s:.4f}s < 0 — t_start after first completion?"
        )
    if n > n_warm:
        steady_s = ordered[-1] - ordered[n_warm - 1]
        steady_n = n - n_warm
        steady_per = steady_s / steady_n
    else:
        steady_s = 0.0
        steady_n = 0
        steady_per = None
    blended = (ordered[-1] - t_start) / n
    return {
        "n_items": n,
        "n_warmup": n_warm,
        "warmup_s": warmup_s,
        "steady_n": steady_n,
        "steady_s": steady_s,
        "steady_per_item": steady_per,
        "blended_per_item": blended,
    }


def hours(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return seconds / 3600.0
