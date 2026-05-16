"""Tests for the multi-method statistical layer.

Each method is run against synthetic distributions where the right answer is known.
Then `consensus` is tested for agreement under (a) clear win, (b) clear noise,
(c) borderline case.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.stats.methods import (
    cliffs_delta,
    consensus,
    hedges_g,
    paired_t_test,
    permutation_test,
    wilcoxon_signed_rank,
)


def _seed(s: int = 42):
    return np.random.default_rng(s)


def _paired_with_lift(lift: float, n: int = 20, sigma: float = 0.2, seed: int = 0):
    """Return (baseline, variant) where variant systematically beats baseline by `lift`."""
    rng = _seed(seed)
    base = rng.uniform(0.4, 0.7, size=n)
    var = np.clip(base + lift + rng.normal(0, sigma, size=n), 0, 1)
    return base.tolist(), var.tolist()


# ────────────────────────────── individual tests


def test_paired_t_significant_on_clear_lift():
    base, var = _paired_with_lift(lift=0.30, n=20, sigma=0.05)
    r = paired_t_test(base, var)
    assert r.p_value < 0.01
    assert r.statistic > 0


def test_paired_t_not_significant_on_noise():
    base, var = _paired_with_lift(lift=0.0, n=20, sigma=0.2)
    r = paired_t_test(base, var)
    assert r.p_value > 0.05


def test_wilcoxon_significant_on_clear_lift():
    base, var = _paired_with_lift(lift=0.30, n=20, sigma=0.05)
    r = wilcoxon_signed_rank(base, var)
    assert r.p_value < 0.01


def test_wilcoxon_not_significant_on_noise():
    base, var = _paired_with_lift(lift=0.0, n=20, sigma=0.2)
    r = wilcoxon_signed_rank(base, var)
    assert r.p_value > 0.05


def test_permutation_significant_on_clear_lift():
    base, var = _paired_with_lift(lift=0.30, n=20, sigma=0.05)
    r = permutation_test(base, var, n_permutations=5_000)
    assert r.p_value < 0.05
    assert r.statistic > 0


def test_permutation_not_significant_on_noise():
    base, var = _paired_with_lift(lift=0.0, n=20, sigma=0.2)
    r = permutation_test(base, var, n_permutations=5_000)
    assert r.p_value > 0.05


def test_hedges_g_large_lift():
    base, var = _paired_with_lift(lift=0.40, n=20, sigma=0.05)
    e = hedges_g(base, var)
    assert e.value > 0.8
    assert e.interpretation == "large"


def test_hedges_g_negligible_on_noise():
    base, var = _paired_with_lift(lift=0.0, n=20, sigma=0.2)
    e = hedges_g(base, var)
    assert abs(e.value) < 0.3


def test_cliffs_delta_large_lift():
    base, var = _paired_with_lift(lift=0.40, n=20, sigma=0.05)
    e = cliffs_delta(base, var)
    assert e.value > 0.6
    assert e.interpretation == "large"


def test_cliffs_delta_negligible_on_noise():
    base, var = _paired_with_lift(lift=0.0, n=30, sigma=0.2)
    e = cliffs_delta(base, var)
    assert abs(e.value) < 0.4


# ────────────────────────────── consensus


def test_consensus_adopts_clear_winner():
    base, var = _paired_with_lift(lift=0.30, n=20, sigma=0.05)
    mv = consensus(base, var)
    assert mv.sig_votes_positive == mv.sig_total
    assert mv.effect_votes_meaningful == mv.effect_total
    assert mv.adopt()


def test_consensus_rejects_noise():
    base, var = _paired_with_lift(lift=0.0, n=20, sigma=0.2)
    mv = consensus(base, var)
    assert mv.sig_votes_positive == 0
    assert not mv.adopt()


def test_consensus_borderline_split():
    """A weak lift produces partial agreement — significance tests split, effect
    sizes are small/negligible. Should NOT adopt under default 2/3 thresholds."""
    base, var = _paired_with_lift(lift=0.05, n=15, sigma=0.20, seed=7)
    mv = consensus(base, var)
    assert not mv.adopt()


def test_consensus_records_all_methods():
    base, var = _paired_with_lift(lift=0.30, n=20, sigma=0.05)
    mv = consensus(base, var)
    sig_names = {r.name for r in mv.sig_results}
    eff_names = {e.name for e in mv.effect_results}
    assert sig_names == {"paired_t_test", "wilcoxon", "permutation"}
    assert eff_names == {"cohens_d", "hedges_g", "cliffs_delta"}
