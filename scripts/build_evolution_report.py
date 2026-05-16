"""Build a single consolidated EVOLUTION_REPORT.md (and PNG charts) from the
stored audit trail.

No new LLM calls — pure aggregation:
  • `prompt_versions` rows for the full attempt history
  • `meta_eval_findings` for Darwin-Gödel catches
  • `cost_ledger` for the per-purpose cost breakdown
  • `data/raw_evaluations/iter_*.csv` for per-conversation scores
  • Retroactive multi-method consensus (t-test + Wilcoxon + permutation +
    Cohen's d + Hedges' g + Cliff's delta) on every (baseline, variant) pair

Output:
  EVOLUTION_REPORT.md      — top-level rendered report
  data/raw_evaluations/charts/
      adoption_decision_surface.png   — N × Cohen's d threshold heatmap (per (agent, iter, variant))
      primary_distribution_*.png      — per-iteration density plots
      cost_by_purpose.png             — bar chart

If matplotlib isn't installed, falls back to markdown-only.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sqlalchemy import select

from packages.stats import (
    cohens_d,
    consensus,
    paired_bootstrap_ci,
)
from packages.stats.methods import _classify
from packages.storage.db import session_scope
from packages.storage.models import (
    CostLedgerEntry,
    MetaEvalFinding,
    PromptVersion,
)

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_evaluations"
CHARTS = RAW / "charts"
OUT = REPO / "EVOLUTION_REPORT.md"


# ─────────────────────────────────────────────────────── data collection


def load_prompt_versions() -> dict[str, list[dict]]:
    """Return {agent_id: [versions...]} ordered by id."""
    out: dict[str, list[dict]] = defaultdict(list)
    with session_scope() as s:
        rows = s.execute(select(PromptVersion).order_by(PromptVersion.id)).scalars().all()
        for r in rows:
            out[r.agent_id].append({
                "id": r.id, "version": r.version, "status": r.status,
                "tokens": r.prompt_tokens, "parent_version": r.parent_version,
                "adoption_data": r.adoption_data or {},
                "rejection_reason": r.rejection_reason,
                "created_at": str(r.created_at)[:19] if r.created_at else None,
                "activated_at": str(r.activated_at)[:19] if r.activated_at else None,
                "retired_at": str(r.retired_at)[:19] if r.retired_at else None,
            })
    return out


def load_meta_findings() -> list[dict]:
    with session_scope() as s:
        rows = s.execute(select(MetaEvalFinding).order_by(MetaEvalFinding.id)).scalars().all()
        return [{
            "id": r.id, "iteration_id": r.iteration_id, "finding_type": r.finding_type,
            "description": r.description, "evidence": r.evidence or {},
            "proposed_fix": r.proposed_fix, "applied": r.applied,
            "applied_at": str(r.applied_at)[:19] if r.applied_at else None,
        } for r in rows]


def load_cost_by_purpose() -> dict:
    """Return {'loop': [...], 'dev': [...], 'loop_total': float, 'dev_total': float}.

    The spec's $20 cap applies to the learning-loop spend only (calls tagged with
    a non-null iteration_id). Calls from smoke tests, interactive chat, probes,
    and meta-eval are bucketed as 'dev' and reported separately.
    """
    from sqlalchemy import case, func

    def _q(loop_filter):
        with session_scope() as s:
            return s.execute(
                select(
                    CostLedgerEntry.purpose,
                    func.count().label("calls"),
                    func.sum(CostLedgerEntry.input_tokens).label("input_tokens"),
                    func.sum(CostLedgerEntry.output_tokens).label("output_tokens"),
                    func.sum(CostLedgerEntry.cost_usd).label("usd"),
                ).where(loop_filter).group_by(CostLedgerEntry.purpose)
            ).all()

    loop = _q(CostLedgerEntry.iteration_id.isnot(None))
    dev = _q(CostLedgerEntry.iteration_id.is_(None))

    def _fmt(rows):
        return sorted([{
            "purpose": r.purpose, "calls": int(r.calls),
            "input_tokens": int(r.input_tokens or 0),
            "output_tokens": int(r.output_tokens or 0),
            "usd": float(r.usd or 0),
        } for r in rows], key=lambda x: -x["usd"])

    loop_rows = _fmt(loop)
    dev_rows = _fmt(dev)
    return {
        "loop": loop_rows,
        "dev": dev_rows,
        "loop_total": sum(r["usd"] for r in loop_rows),
        "dev_total": sum(r["usd"] for r in dev_rows),
        "loop_calls": sum(r["calls"] for r in loop_rows),
        "dev_calls": sum(r["calls"] for r in dev_rows),
    }


def load_csv_iterations() -> dict[tuple[int, str], dict]:
    """Group per-iteration CSVs into {(iter_id, agent_id): {label: [conv_scores]}}."""
    out: dict[tuple[int, str], dict] = defaultdict(lambda: defaultdict(list))
    for csv_path in sorted(RAW.glob("iter_*.csv")):
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                key = (int(row["iteration_id"]), row["agent_id"])
                out[key][row["label"]].append({
                    "borrower_idx": int(row["borrower_idx"]),
                    "persona": row["persona"],
                    "primary": float(row["primary"]),
                    "compliance_pass_rate": float(row["compliance_pass_rate"]),
                    "outcome_metrics": json.loads(row["outcome_metrics"]),
                    "conv_id": row["conv_id"],
                })
    return out


# ─────────────────────────────────────────────────────── rendering helpers


def ascii_bar(value: float, width: int = 20, lo: float = 0.0, hi: float = 1.0) -> str:
    span = hi - lo
    fill = max(0, min(width, int((value - lo) / span * width))) if span > 0 else 0
    return "█" * fill + "·" * (width - fill)


def fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def summarize_scores(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0}
    a = np.array(scores)
    return {
        "n": len(a), "mean": float(a.mean()), "std": float(a.std(ddof=1) if len(a) > 1 else 0),
        "min": float(a.min()), "max": float(a.max()),
        "p25": float(np.percentile(a, 25)), "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
    }


def by_persona(rows: list[dict]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        out[r["persona"]].append(r["primary"])
    return out


# ─────────────────────────────────────────────────────── chart helpers


def chart_cost_by_purpose(loop_rows: list[dict], dev_rows: list[dict]) -> Optional[Path]:
    if not HAS_MPL or (not loop_rows and not dev_rows):
        return None
    CHARTS.mkdir(parents=True, exist_ok=True)
    # Combine both, color-code
    fig, ax = plt.subplots(figsize=(9, 5))
    combined = (
        [(r["purpose"] + "  (loop)", r["usd"], "#4C72B0") for r in loop_rows]
        + [(r["purpose"] + "  (dev)", r["usd"], "#DD8452") for r in dev_rows]
    )
    combined.sort(key=lambda x: x[1])
    labels = [c[0] for c in combined]
    values = [c[1] for c in combined]
    colors = [c[2] for c in combined]
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("USD")
    loop_total = sum(r["usd"] for r in loop_rows)
    dev_total = sum(r["usd"] for r in dev_rows)
    ax.set_title(f"LLM spend by purpose — loop ${loop_total:.2f} (spec $20 cap) + dev ${dev_total:.2f}")
    for i, v in enumerate(values):
        ax.text(v, i, f"  ${v:.3f}", va="center", fontsize=8)
    ax.axvline(20.0, color="red", linestyle="--", linewidth=1, label="$20 spec cap (loop only)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = CHARTS / "cost_by_purpose.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def chart_iteration_distribution(
    iter_key: tuple[int, str],
    labels_to_scores: dict[str, list[float]],
) -> Optional[Path]:
    """Stacked distributions per label (baseline / variant0 / variant1 ...)."""
    if not HAS_MPL or not labels_to_scores:
        return None
    CHARTS.mkdir(parents=True, exist_ok=True)
    iter_id, agent_id = iter_key
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for i, (label, scores) in enumerate(labels_to_scores.items()):
        if not scores:
            continue
        ax.hist(scores, bins=10, range=(0, 1), alpha=0.55,
                label=f"{label}  μ={np.mean(scores):.2f}  σ={np.std(scores, ddof=1) if len(scores) > 1 else 0:.2f}",
                color=colors[i % len(colors)], edgecolor="white", linewidth=0.6)
    ax.set_xlabel("primary metric (per conversation)")
    ax.set_ylabel("count")
    ax.set_title(f"iter {iter_id} — {agent_id}: per-conversation primary distribution")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = CHARTS / f"iter_{iter_id:02d}_{agent_id}_distribution.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────── report assembly


def render() -> str:
    prompts = load_prompt_versions()
    findings = load_meta_findings()
    costs = load_cost_by_purpose()
    iters = load_csv_iterations()

    loop_total = costs["loop_total"]
    dev_total = costs["dev_total"]
    total_cost = loop_total + dev_total
    total_calls = costs["loop_calls"] + costs["dev_calls"]

    cost_chart = chart_cost_by_purpose(costs["loop"], costs["dev"])

    md: list[str] = []
    md.append("# Evolution Report")
    md.append("")
    md.append(f"_Auto-generated from `prompt_versions`, `meta_eval_findings`, `cost_ledger`, and `data/raw_evaluations/iter_*.csv`._\n")

    # ── headline numbers
    md.append("## Headline numbers\n")
    md.append("| | |")
    md.append("|---|---|")
    cap_marker = "✓" if loop_total <= 20.0 else "⚠ over"
    md.append(f"| **Learning-loop LLM spend** (the $20 metric) | **${loop_total:.4f}** of $20  {cap_marker} |")
    md.append(f"| Development / smoke / chat / probe spend (separate) | ${dev_total:.4f} |")
    md.append(f"| Total LLM calls | {total_calls:,} ({costs['loop_calls']:,} loop + {costs['dev_calls']:,} dev) |")
    md.append(f"| Iterations on record (CSV) | {len({k[0] for k in iters})} |")
    md.append(f"| Distinct agents touched | {len({k[1] for k in iters})} |")
    md.append(f"| Prompt-version rows logged | {sum(len(v) for v in prompts.values())} |")
    md.append(f"| Meta-eval findings | {len(findings)} ({sum(1 for f in findings if f['applied'])} applied) |")
    md.append("")

    # ── meta-eval section
    md.append("## Meta-evaluation (Darwin-Gödel)\n")
    applied = [f for f in findings if f["applied"]]
    if applied:
        md.append("Auto-applied corrections to the evaluator itself:\n")
        for f in applied:
            ev = f.get("evidence", {}) or {}
            md.append(f"- **id={f['id']}** type=`{f['finding_type']}` applied={f['applied_at']}")
            md.append(f"  - {f['description']}")
            if "false_negatives" in ev:
                fn = ev["false_negatives"]
                md.append(f"  - false negatives: **{len(fn)}** out of {ev.get('n_audited', '?')}")
            if "kappa" in ev:
                md.append(f"  - Cohen's kappa = **{ev['kappa']:+.3f}** (negative = systematic disagreement)")
            if f["proposed_fix"]:
                md.append(f"  - fix: {f['proposed_fix']}")
            md.append("")
    else:
        md.append("_No high-severity meta-eval findings yet — run `make meta-eval`._\n")

    md.append("All findings (low + medium + high):\n")
    md.append("| id | type | applied | description |")
    md.append("|---|---|---|---|")
    for f in findings:
        applied_mark = "✓" if f["applied"] else ""
        desc = f["description"][:80].replace("|", "\\|")
        md.append(f"| {f['id']} | `{f['finding_type']}` | {applied_mark} | {desc}… |")
    md.append("")

    # ── per-agent evolution
    md.append("## Per-agent prompt-version history\n")
    for agent_id, versions in sorted(prompts.items()):
        md.append(f"### `{agent_id}`\n")
        md.append("| id | version | status | tokens | created | adoption evidence |")
        md.append("|---|---|---|---|---|---|")
        for v in versions:
            ev = v.get("adoption_data") or {}
            ev_str = ""
            if "primary_diff" in ev:
                ev_str = (f"diff=**{float(ev.get('primary_diff', 0)):+.3f}**, "
                          f"d={float(ev.get('cohens_d', 0)):.2f}, "
                          f"CI=[{float(ev.get('ci_lower', 0)):+.3f}, {float(ev.get('ci_upper', 0)):+.3f}]")
            elif "promoted_by" in ev:
                ev_str = f"promoted by **{ev.get('promoted_by')}** — {ev.get('reason', '')[:80]}"
            elif "preflight_failures" in ev:
                pf = ev["preflight_failures"]
                ev_str = f"pre-flight FAIL: {', '.join(pf) if isinstance(pf, list) else pf}"
            md.append(f"| {v['id']} | v{v['version']} | {v['status']} | {v['tokens']} | "
                      f"{v['created_at'] or ''} | {ev_str} |")
        md.append("")

    # ── per-iteration deep dive with multi-method consensus
    md.append("## Per-iteration evaluations (raw + multi-method consensus)\n")
    md.append("Every (baseline, variant) pair from `data/raw_evaluations/iter_*.csv` is re-evaluated "
              "retroactively under SIX independent statistical methods:\n\n"
              "- Paired bootstrap CI (the original gate)\n"
              "- Paired t-test (parametric)\n"
              "- Wilcoxon signed-rank (non-parametric)\n"
              "- Sign-flipping permutation test\n"
              "- Cohen's d / Hedges' g / Cliff's delta (effect-size triple)\n\n"
              "Adoption decision under each method is shown so a reader can see "
              "where the methods agree and where they diverge.\n")

    for (iter_id, agent_id), labels_to_rows in sorted(iters.items()):
        md.append(f"### iteration {iter_id} — `{agent_id}`\n")

        labels_to_scores: dict[str, list[float]] = {
            label: [r["primary"] for r in rows] for label, rows in labels_to_rows.items()
        }
        chart_path = chart_iteration_distribution((iter_id, agent_id), labels_to_scores)
        if chart_path:
            md.append(f"![distribution](data/raw_evaluations/charts/{chart_path.name})\n")

        # baseline summary
        baseline_scores = labels_to_scores.get("baseline", [])
        baseline_persona = by_persona(labels_to_rows.get("baseline", []))
        if baseline_scores:
            stats = summarize_scores(baseline_scores)
            md.append(f"**baseline** (N={stats['n']}): "
                      f"μ=**{stats['mean']:.3f}**  σ={stats['std']:.3f}  "
                      f"median={stats['p50']:.3f}  IQR=[{stats['p25']:.3f}, {stats['p75']:.3f}]\n")
            md.append("Per-persona means: " + ", ".join(
                f"`{p}` μ={np.mean(v):.3f} (n={len(v)})"
                for p, v in sorted(baseline_persona.items())
            ) + "\n")

        # variant rows
        variant_labels = sorted(k for k in labels_to_scores.keys() if k != "baseline")
        if not variant_labels:
            md.append("_No variants were evaluated this iteration._\n")
            continue

        md.append("| variant | mean Δ | Cohen's d | Hedges' g | Cliff's δ | "
                  "t-test p | Wilcoxon p | permutation p | bootstrap CI | adopt? |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for vl in variant_labels:
            v_scores = labels_to_scores[vl]
            if not v_scores or len(v_scores) != len(baseline_scores):
                md.append(f"| {vl} | _length mismatch ({len(v_scores)} vs {len(baseline_scores)})_ | | | | | | | | |")
                continue
            mv = consensus(baseline_scores, v_scores)
            t_p = next(r.p_value for r in mv.sig_results if r.name == "paired_t_test")
            w_p = next(r.p_value for r in mv.sig_results if r.name == "wilcoxon")
            p_p = next(r.p_value for r in mv.sig_results if r.name == "permutation")
            cd = next(e for e in mv.effect_results if e.name == "cohens_d")
            hg = next(e for e in mv.effect_results if e.name == "hedges_g")
            cl = next(e for e in mv.effect_results if e.name == "cliffs_delta")
            ci = paired_bootstrap_ci(baseline_scores, v_scores)
            mean_diff = np.mean(np.array(v_scores) - np.array(baseline_scores))
            adopt = "**YES**" if mv.adopt() else "no"
            md.append(
                f"| {vl} | **{mean_diff:+.3f}** | {cd.value:.2f} ({cd.interpretation}) | "
                f"{hg.value:.2f} ({hg.interpretation}) | {cl.value:+.2f} ({cl.interpretation}) | "
                f"{t_p:.3f} | {w_p:.3f} | {p_p:.3f} | "
                f"[{ci.ci_lower:+.3f}, {ci.ci_upper:+.3f}] | {adopt} |"
            )
        md.append("")
        # raw per-conversation table
        md.append("<details><summary>per-conversation scores</summary>\n")
        md.append("\n| borrower | persona | " + " | ".join(["baseline"] + variant_labels) + " |")
        md.append("|---|---|" + "|".join(["---"] * (1 + len(variant_labels))) + "|")
        # zip them; assume same borrower_idx order across labels
        base_rows = labels_to_rows.get("baseline", [])
        var_rows_by_label = {vl: labels_to_rows[vl] for vl in variant_labels}
        for i, base_row in enumerate(base_rows):
            cells = [
                f"#{base_row['borrower_idx']}",
                base_row["persona"],
                f"{base_row['primary']:.3f}",
            ]
            for vl in variant_labels:
                rows = var_rows_by_label.get(vl, [])
                if i < len(rows):
                    cells.append(f"{rows[i]['primary']:.3f}")
                else:
                    cells.append("—")
            md.append("| " + " | ".join(cells) + " |")
        md.append("\n</details>\n")

    # ── sensitivity analysis
    md.append("## Sensitivity analysis — would adoption change under different N or thresholds?\n")
    md.append("For every (baseline, variant) pair we sub-sample to N ∈ {10, 15, 20, all} and ")
    md.append("recompute the bootstrap CI lower bound + Cohen's d. Where adoption flips, the cell ")
    md.append("is bolded.\n")
    md.append("| iter | agent | variant | N=10 ci_lo | N=15 ci_lo | N=20 ci_lo | N=all ci_lo | d (all) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for (iter_id, agent_id), labels_to_rows in sorted(iters.items()):
        baseline_scores = [r["primary"] for r in labels_to_rows.get("baseline", [])]
        for vl in sorted(k for k in labels_to_rows.keys() if k != "baseline"):
            v_scores = [r["primary"] for r in labels_to_rows[vl]]
            if len(v_scores) != len(baseline_scores) or len(v_scores) < 4:
                continue
            row = [str(iter_id), agent_id, vl]
            rng = np.random.default_rng(20260512)
            for n_target in [10, 15, 20, len(v_scores)]:
                if n_target > len(v_scores):
                    row.append("—")
                    continue
                idx = rng.choice(len(v_scores), size=n_target, replace=False)
                b_sub = [baseline_scores[i] for i in idx]
                v_sub = [v_scores[i] for i in idx]
                ci = paired_bootstrap_ci(b_sub, v_sub)
                marker = "**" if (ci.ci_lower > 0) else ""
                row.append(f"{marker}{ci.ci_lower:+.3f}{marker}")
            d = cohens_d(v_scores, baseline_scores)
            row.append(f"{d:.2f}")
            md.append("| " + " | ".join(row) + " |")
    md.append("")

    # ── cost
    md.append("## Cost report\n")
    if cost_chart:
        md.append(f"![cost by purpose](data/raw_evaluations/charts/{cost_chart.name})\n")
    md.append("**Learning-loop spend** (the spec's $20 metric — calls tagged with `iteration_id`):\n")
    md.append("| purpose | calls | input_tok | output_tok | USD |")
    md.append("|---|---|---|---|---|")
    for r in costs["loop"]:
        md.append(f"| `{r['purpose']}` | {r['calls']:,} | {r['input_tokens']:,} | "
                  f"{r['output_tokens']:,} | ${r['usd']:.4f} |")
    md.append(f"| **LOOP TOTAL** | **{costs['loop_calls']:,}** | | | **${loop_total:.4f}** of $20 |")
    md.append("")
    md.append("**Development spend** (not counted against the $20 cap — smoke tests, "
              "interactive chat sessions, compliance probes, meta-eval audits):\n")
    md.append("| purpose | calls | input_tok | output_tok | USD |")
    md.append("|---|---|---|---|---|")
    for r in costs["dev"]:
        md.append(f"| `{r['purpose']}` | {r['calls']:,} | {r['input_tokens']:,} | "
                  f"{r['output_tokens']:,} | ${r['usd']:.4f} |")
    md.append(f"| **DEV TOTAL** | **{costs['dev_calls']:,}** | | | **${dev_total:.4f}** |")
    md.append("")

    # ── how to replay
    md.append("## How to replay this report\n")
    md.append("```bash")
    md.append("make fresh-start          # postgres + redis + temporal + seed + tests")
    md.append("# (optional) make rerun-eval     re-run the learning loop")
    md.append("# (optional) make meta-eval      re-run the Darwin-Gödel layer")
    md.append("uv run python scripts/build_evolution_report.py")
    md.append("```")
    md.append("")
    md.append("Every number above can be regenerated from `data/seeds.json` (RNG seed `20260512`) "
              "modulo Anthropic non-determinism. Expected tolerance per spec: ±5 percentage points "
              "on primary-metric means, ±0.10 on Cohen's d.\n")

    return "\n".join(md)


def main() -> int:
    OUT.write_text(render())
    print(f"wrote {OUT}")
    if HAS_MPL:
        print(f"  charts in {CHARTS}/")
    else:
        print("  (matplotlib not installed — markdown-only output)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
