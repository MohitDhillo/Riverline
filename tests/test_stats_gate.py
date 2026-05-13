"""Statistical-gate tests on synthetic distributions.

No LLM calls. Validates the gate makes the right call on:
  - true improvement → adopt
  - tiny noise → reject (Cohen's d too small)
  - mixed effect → reject (CI straddles 0)
  - compliance regression → reject hard
  - system metric regresses → reject
"""

from __future__ import annotations

import numpy as np

from packages.stats import (
    GateDecision,
    cohens_d,
    evaluate_gate,
    non_inferiority_p,
    paired_bootstrap_ci,
)


def _seed():
    return np.random.default_rng(42)


def test_bootstrap_ci_paired_lift_is_positive() -> None:
    rng = _seed()
    base = rng.normal(0.5, 0.1, 30).tolist()
    variant = [b + 0.15 + rng.normal(0, 0.02) for b in base]
    ci = paired_bootstrap_ci(base, variant)
    assert ci.ci_lower > 0
    assert 0.10 < ci.mean_diff < 0.20


def test_bootstrap_ci_no_lift_straddles_zero() -> None:
    rng = _seed()
    base = rng.normal(0.5, 0.1, 30).tolist()
    variant = rng.normal(0.5, 0.1, 30).tolist()
    ci = paired_bootstrap_ci(base, variant)
    assert ci.ci_lower < 0 < ci.ci_upper


def test_cohens_d_large_lift() -> None:
    rng = _seed()
    base = rng.normal(0.4, 0.05, 50).tolist()
    variant = rng.normal(0.7, 0.05, 50).tolist()
    d = cohens_d(variant, base)
    assert d > 1.5  # huge separation


def test_cohens_d_zero_when_identical() -> None:
    base = [0.5] * 10
    variant = [0.5] * 10
    assert cohens_d(variant, base) == 0.0


def test_non_inferiority_p_when_variant_clearly_better() -> None:
    rng = _seed()
    base = rng.normal(0.5, 0.05, 30).tolist()
    variant = [b + 0.10 for b in base]
    # variant > baseline so P(variant - baseline < -margin) is ~0
    assert non_inferiority_p(variant, base, margin=0.05) < 0.05


def test_non_inferiority_p_when_variant_clearly_worse() -> None:
    rng = _seed()
    base = rng.normal(0.5, 0.05, 30).tolist()
    variant = [b - 0.15 for b in base]
    # variant much worse → high p (rejects non-inferiority)
    assert non_inferiority_p(variant, base, margin=0.05) > 0.5


def test_gate_adopts_clear_winner() -> None:
    rng = _seed()
    base = rng.normal(0.55, 0.07, 30).tolist()
    variant = [b + 0.18 + rng.normal(0, 0.02) for b in base]
    r = evaluate_gate(
        baseline_primary=base,
        variant_primary=variant,
        baseline_compliance=[1.0] * 30,
        variant_compliance=[1.0] * 30,
        baseline_system=[1.0, 0.0] * 15,
        variant_system=[1.0, 0.0] * 15,
    )
    assert r.decision == GateDecision.ADOPT, r.reasons


def test_gate_rejects_noise() -> None:
    rng = _seed()
    base = rng.normal(0.6, 0.1, 30).tolist()
    variant = rng.normal(0.6, 0.1, 30).tolist()
    r = evaluate_gate(
        baseline_primary=base,
        variant_primary=variant,
        baseline_compliance=[1.0] * 30,
        variant_compliance=[1.0] * 30,
        baseline_system=[1.0] * 30,
        variant_system=[1.0] * 30,
    )
    assert r.decision in (GateDecision.REJECT_CI, GateDecision.REJECT_EFFECT_SIZE)


def test_gate_rejects_compliance_regression_even_when_primary_better() -> None:
    rng = _seed()
    base = rng.normal(0.5, 0.05, 30).tolist()
    variant = [b + 0.2 for b in base]      # primary is better
    r = evaluate_gate(
        baseline_primary=base,
        variant_primary=variant,
        baseline_compliance=[1.0] * 30,
        variant_compliance=[0.8] * 30,       # but compliance regressed
        baseline_system=[1.0] * 30,
        variant_system=[1.0] * 30,
    )
    assert r.decision == GateDecision.REJECT_COMPLIANCE


def test_gate_rejects_when_system_metric_clearly_worse() -> None:
    rng = _seed()
    base = rng.normal(0.5, 0.05, 30).tolist()
    variant = [b + 0.2 for b in base]      # primary better
    r = evaluate_gate(
        baseline_primary=base,
        variant_primary=variant,
        baseline_compliance=[1.0] * 30,
        variant_compliance=[1.0] * 30,
        baseline_system=[1.0] * 30,
        variant_system=[0.4] * 30,           # system metric tanked
    )
    assert r.decision == GateDecision.REJECT_SYSTEM
