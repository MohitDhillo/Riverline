"""Statistical primitives for the adoption gate.

Per FINAL_PLAN §1, adoption requires ALL of:
  1. paired bootstrap 95% CI of (variant - incumbent) lower bound > 0
  2. Cohen's d > 0.2 (effect-size floor)
  3. compliance pass rate ≥ incumbent (separate hard gate — handled in gate.py)
  4. system-level resolution rate non-inferior at p > 0.10 (handled in gate.py)

These are pure-math helpers. No LLM calls. Easy to test with synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapCI:
    mean_diff: float
    ci_lower: float
    ci_upper: float
    n: int


def paired_bootstrap_ci(
    baseline: list[float],
    variant: list[float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 20260512,
) -> BootstrapCI:
    """Paired-sample bootstrap CI for mean(variant - baseline).

    Requires len(baseline) == len(variant) — same borrowers, paired.
    """
    if len(baseline) != len(variant):
        raise ValueError(f"paired bootstrap requires equal-length samples; "
                         f"got {len(baseline)} vs {len(variant)}")
    if not baseline:
        return BootstrapCI(0.0, 0.0, 0.0, 0)

    diffs = np.array(variant, dtype=float) - np.array(baseline, dtype=float)
    n = len(diffs)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(
        mean_diff=float(diffs.mean()),
        ci_lower=float(lo),
        ci_upper=float(hi),
        n=n,
    )


def cohens_d(variant: list[float], baseline: list[float]) -> float:
    """Standard Cohen's d for paired samples.

    Returns 0 if either sample has zero variance (avoids division by zero —
    no detectable effect on truly constant data).
    """
    a = np.array(variant, dtype=float)
    b = np.array(baseline, dtype=float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 else 0.0
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def non_inferiority_p(
    variant: list[float],
    baseline: list[float],
    *,
    margin: float = 0.05,
) -> float:
    """Bootstrap one-sided p-value for H0: variant_mean < baseline_mean - margin.

    Returns the proportion of bootstrap resamples where variant - baseline < -margin.
    A low p means we can reject "variant is materially worse"; we want p > 0.10
    to consider the variant non-inferior on system-level metrics.
    """
    if not baseline:
        return 1.0
    if len(baseline) != len(variant):
        raise ValueError("non-inferiority test requires paired samples")
    a = np.array(variant, dtype=float)
    b = np.array(baseline, dtype=float)
    rng = np.random.default_rng(20260512)
    n_resamples = 10_000
    n = len(a)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = (a - b)[idx].mean(axis=1)
    worse = (diffs < -margin).mean()
    return float(worse)
