"""Multiple independent statistical methods for paired sample comparison.

Each method gives a *significance signal* and an *effect-size signal*. We run all
of them on the same paired (baseline, variant) data and report agreement —
so the adoption decision isn't hostage to one method's assumptions.

  Significance tests              Effect-size measures
  -------------------------       ---------------------
  paired_t_test  (parametric)     cohens_d       (parametric, in stats.tests)
  wilcoxon       (rank-based)     hedges_g       (small-N corrected Cohen's d)
  permutation    (non-parametric) cliffs_delta   (non-parametric)
  bootstrap CI   (already in tests.py)

`consensus()` combines them into a single MajorityVote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats as scipy_stats


# ────────────────────────────────────────────────────────── significance tests


@dataclass
class TestResult:
    name: str
    p_value: float
    statistic: float
    one_sided: bool = True

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def paired_t_test(baseline: list[float], variant: list[float],
                  alternative: Literal["greater", "two-sided"] = "greater") -> TestResult:
    """Paired-sample t-test on (variant - baseline).

    Default alternative is one-sided 'greater' (we adopt only when variant > baseline).
    Assumes the paired differences are approximately normal — fine when N >= 12 ish
    and there are no extreme outliers. Compared against Wilcoxon for robustness.
    """
    if len(baseline) != len(variant):
        raise ValueError("paired t-test requires equal-length samples")
    if len(baseline) < 2:
        return TestResult("paired_t_test", 1.0, 0.0, one_sided=alternative != "two-sided")
    res = scipy_stats.ttest_rel(variant, baseline, alternative=alternative)
    return TestResult(
        name="paired_t_test",
        p_value=float(res.pvalue),
        statistic=float(res.statistic),
        one_sided=alternative != "two-sided",
    )


def wilcoxon_signed_rank(baseline: list[float], variant: list[float],
                         alternative: Literal["greater", "two-sided"] = "greater") -> TestResult:
    """Wilcoxon signed-rank — non-parametric paired analogue of the t-test.

    Doesn't assume normality. Pairs with zero difference are dropped (default in
    scipy). Useful when the primary metric has many ties or a non-normal
    distribution — both true for our [0..1] composite.
    """
    if len(baseline) != len(variant):
        raise ValueError("wilcoxon requires equal-length samples")
    diffs = np.array(variant) - np.array(baseline)
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 1:
        return TestResult("wilcoxon", 1.0, 0.0, one_sided=alternative != "two-sided")
    try:
        res = scipy_stats.wilcoxon(variant, baseline, alternative=alternative,
                                    zero_method="wilcox", correction=False)
        return TestResult(
            name="wilcoxon",
            p_value=float(res.pvalue),
            statistic=float(res.statistic),
            one_sided=alternative != "two-sided",
        )
    except ValueError:
        return TestResult("wilcoxon", 1.0, 0.0, one_sided=alternative != "two-sided")


def permutation_test(baseline: list[float], variant: list[float],
                     n_permutations: int = 10_000,
                     rng_seed: int = 20260512) -> TestResult:
    """Sign-flipping permutation test on paired differences.

    Under the null hypothesis (variant = baseline), the sign of each (variant -
    baseline) is exchangeable. We compute the observed mean diff, then for each
    permutation flip each sign at random and compute the resampled mean. The
    p-value is the proportion of permutations where the resampled mean is
    >= observed.

    Most assumption-free of the three. Slower but cheap at N=15.
    """
    if len(baseline) != len(variant):
        raise ValueError("permutation requires equal-length samples")
    diffs = np.array(variant) - np.array(baseline)
    n = len(diffs)
    if n == 0:
        return TestResult("permutation", 1.0, 0.0, one_sided=True)
    observed = float(diffs.mean())
    rng = np.random.default_rng(rng_seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    resampled = (signs * np.abs(diffs)).mean(axis=1)
    p = float((resampled >= observed).mean())
    return TestResult(
        name="permutation",
        p_value=p,
        statistic=observed,
        one_sided=True,
    )


# ────────────────────────────────────────────────────────── effect-size measures


@dataclass
class EffectSize:
    name: str
    value: float
    interpretation: str  # "negligible" | "small" | "medium" | "large"


def hedges_g(baseline: list[float], variant: list[float]) -> EffectSize:
    """Hedges' g — Cohen's d corrected for small-sample bias.

    For N < 20 (which we live in), Cohen's d overestimates. Hedges multiplies
    by a correction factor that approaches 1 as N grows.
    """
    a = np.array(variant, dtype=float)
    b = np.array(baseline, dtype=float)
    n = len(a)
    if n < 2:
        return EffectSize("hedges_g", 0.0, "negligible")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled == 0:
        return EffectSize("hedges_g", 0.0, "negligible")
    d = (a.mean() - b.mean()) / pooled
    # Hedges small-sample correction
    correction = 1 - 3 / (4 * (2 * n) - 9) if (2 * n) > 9 else 1.0
    g = float(d * correction)
    return EffectSize("hedges_g", g, _classify(abs(g)))


def cliffs_delta(baseline: list[float], variant: list[float]) -> EffectSize:
    """Cliff's delta — non-parametric effect size.

    P(variant > baseline) - P(variant < baseline). Ranges [-1, 1]. Doesn't assume
    normality or paired structure. Conventional thresholds: |δ| < 0.147 negligible,
    < 0.33 small, < 0.474 medium, otherwise large.
    """
    a = np.array(variant, dtype=float)
    b = np.array(baseline, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return EffectSize("cliffs_delta", 0.0, "negligible")
    greater = sum(x > y for x in a for y in b)
    less = sum(x < y for x in a for y in b)
    n_pairs = len(a) * len(b)
    delta = float((greater - less) / n_pairs) if n_pairs else 0.0
    return EffectSize("cliffs_delta", delta, _classify_cliffs(abs(delta)))


def _classify(d_abs: float) -> str:
    if d_abs < 0.2:
        return "negligible"
    if d_abs < 0.5:
        return "small"
    if d_abs < 0.8:
        return "medium"
    return "large"


def _classify_cliffs(d_abs: float) -> str:
    if d_abs < 0.147:
        return "negligible"
    if d_abs < 0.33:
        return "small"
    if d_abs < 0.474:
        return "medium"
    return "large"


# ────────────────────────────────────────────────────────── consensus


@dataclass
class MajorityVote:
    """Agreement summary across all significance + effect-size methods."""

    sig_results: list[TestResult] = field(default_factory=list)
    effect_results: list[EffectSize] = field(default_factory=list)
    sig_votes_positive: int = 0   # number of sig tests with p < alpha
    sig_total: int = 0
    effect_votes_meaningful: int = 0  # effect_size NOT negligible
    effect_total: int = 0

    @property
    def sig_agreement(self) -> float:
        return self.sig_votes_positive / self.sig_total if self.sig_total else 0.0

    @property
    def effect_agreement(self) -> float:
        return self.effect_votes_meaningful / self.effect_total if self.effect_total else 0.0

    def adopt(self, sig_threshold: float = 2 / 3, effect_threshold: float = 2 / 3) -> bool:
        """Adopt iff a supermajority of significance tests AND a supermajority of
        effect-size measures agree the variant is meaningfully better."""
        return (self.sig_agreement >= sig_threshold
                and self.effect_agreement >= effect_threshold)


def consensus(baseline: list[float], variant: list[float],
              *,
              alpha: float = 0.05) -> MajorityVote:
    """Run all 3 significance tests + 2 effect-size measures, return MajorityVote."""
    from packages.stats.tests import cohens_d  # avoid circular import on module load

    sig = [
        paired_t_test(baseline, variant),
        wilcoxon_signed_rank(baseline, variant),
        permutation_test(baseline, variant),
    ]
    cd_val = cohens_d(variant, baseline)
    eff = [
        EffectSize("cohens_d", cd_val, _classify(abs(cd_val))),
        hedges_g(baseline, variant),
        cliffs_delta(baseline, variant),
    ]
    pos = sum(1 for r in sig if r.significant(alpha) and r.statistic > 0)
    meaningful = sum(1 for e in eff if e.value > 0 and e.interpretation != "negligible")
    return MajorityVote(
        sig_results=sig,
        effect_results=eff,
        sig_votes_positive=pos,
        sig_total=len(sig),
        effect_votes_meaningful=meaningful,
        effect_total=len(eff),
    )
