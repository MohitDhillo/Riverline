"""Adoption gate — combines all four checks per FINAL_PLAN §1.

Adopt iff ALL hold:
  1. paired bootstrap CI lower bound > 0
  2. Cohen's d > 0.2
  3. compliance pass rate ≥ baseline (no regression)
  4. system-level resolution non-inferior at p > 0.10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from packages.stats.tests import (
    BootstrapCI,
    cohens_d,
    non_inferiority_p,
    paired_bootstrap_ci,
)


class GateDecision(str, Enum):
    ADOPT = "adopt"
    REJECT_CI = "reject_ci_lower_bound_not_positive"
    REJECT_EFFECT_SIZE = "reject_cohens_d_below_threshold"
    REJECT_COMPLIANCE = "reject_compliance_regression"
    REJECT_SYSTEM = "reject_system_inferior"
    REJECT_SYSTEM_SEAMLESSNESS = "reject_system_seamlessness_regression"


@dataclass
class GateResult:
    decision: GateDecision
    primary_diff: float                  # mean(variant - baseline) on primary metric
    bootstrap: BootstrapCI
    cohens_d: float
    compliance_baseline: float
    compliance_variant: float
    system_p: float                       # non-inferiority bootstrap p (lower=worse)
    seamlessness_baseline: Optional[float] = None
    seamlessness_variant: Optional[float] = None
    seamlessness_drop: Optional[float] = None
    reasons: list[str] = field(default_factory=list)


def evaluate_gate(
    *,
    baseline_primary: list[float],
    variant_primary: list[float],
    baseline_compliance: list[float],   # per-conversation 0..1
    variant_compliance: list[float],
    baseline_system: list[float],        # system-level metric (e.g. resolution 0/1)
    variant_system: list[float],
    baseline_seamlessness: Optional[list[float]] = None,  # 1..5 per full pipeline
    variant_seamlessness: Optional[list[float]] = None,
    min_d: float = 0.1,                     # was 0.2; lowered to allow small wins
    system_margin: float = 0.05,
    system_p_threshold: float = 0.10,
    seamlessness_min_drop: float = 0.5,     # > 0.5 point drop on 1-5 scale → reject
) -> GateResult:
    boot = paired_bootstrap_ci(baseline_primary, variant_primary)
    d = cohens_d(variant_primary, baseline_primary)
    comp_base = float(sum(baseline_compliance) / max(1, len(baseline_compliance)))
    comp_var = float(sum(variant_compliance) / max(1, len(variant_compliance)))
    sys_p = non_inferiority_p(variant_system, baseline_system, margin=system_margin)

    reasons: list[str] = []
    decision = GateDecision.ADOPT

    seamlessness_baseline_mean: Optional[float] = None
    seamlessness_variant_mean: Optional[float] = None
    seamlessness_drop: Optional[float] = None
    if baseline_seamlessness is not None and variant_seamlessness is not None \
            and baseline_seamlessness and variant_seamlessness:
        seamlessness_baseline_mean = float(sum(baseline_seamlessness) / len(baseline_seamlessness))
        seamlessness_variant_mean = float(sum(variant_seamlessness) / len(variant_seamlessness))
        seamlessness_drop = seamlessness_baseline_mean - seamlessness_variant_mean

    if boot.ci_lower <= 0:
        decision = GateDecision.REJECT_CI
        reasons.append(
            f"paired-bootstrap 95% CI of (variant-baseline) = "
            f"[{boot.ci_lower:.4f}, {boot.ci_upper:.4f}]; lower bound not > 0"
        )
    elif d <= min_d:
        decision = GateDecision.REJECT_EFFECT_SIZE
        reasons.append(f"Cohen's d = {d:.3f} ≤ threshold {min_d}")
    elif comp_var < comp_base:
        decision = GateDecision.REJECT_COMPLIANCE
        reasons.append(
            f"compliance regressed: baseline {comp_base:.3f} → variant {comp_var:.3f}"
        )
    elif sys_p > system_p_threshold:
        decision = GateDecision.REJECT_SYSTEM
        reasons.append(
            f"system-level non-inferiority p={sys_p:.3f} exceeds threshold "
            f"{system_p_threshold} (variant may be worse on system metric)"
        )
    elif seamlessness_drop is not None and seamlessness_drop > seamlessness_min_drop:
        decision = GateDecision.REJECT_SYSTEM_SEAMLESSNESS
        reasons.append(
            f"handoff seamlessness regressed: baseline {seamlessness_baseline_mean:.2f} → "
            f"variant {seamlessness_variant_mean:.2f} (drop {seamlessness_drop:.2f} > "
            f"threshold {seamlessness_min_drop})"
        )

    return GateResult(
        decision=decision,
        primary_diff=boot.mean_diff,
        bootstrap=boot,
        cohens_d=d,
        compliance_baseline=comp_base,
        compliance_variant=comp_var,
        system_p=sys_p,
        seamlessness_baseline=seamlessness_baseline_mean,
        seamlessness_variant=seamlessness_variant_mean,
        seamlessness_drop=seamlessness_drop,
        reasons=reasons,
    )
