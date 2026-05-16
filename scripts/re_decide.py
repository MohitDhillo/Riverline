"""Retroactively re-evaluate a stored (baseline, variant) pair from CSV using
the new multi-method consensus gate. If the variant adopts, write a new active
PromptVersion row and lift it to disk as ``prompts/<agent_id>/v0NNN.md``.

No new LLM calls — pure replay of stored per-conversation scores.

Usage:
    uv run python scripts/re_decide.py --agent agent_2 --iter 1 --variant variant1

Prints the full multi-method decision surface, the rationale for the original
proposal, and (if adopted) the new active version + file path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from packages.stats import (
    cohens_d,
    consensus,
    paired_bootstrap_ci,
)
from packages.storage.db import session_scope
from packages.storage.models import PromptVersion
from packages.storage.repos import set_active_prompt, upsert_prompt_version

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_evaluations"


def load_pair(agent_id: str, iter_id: int, variant_label: str) -> tuple[list[float], list[float], list[str]]:
    csv_path = RAW / f"iter_{iter_id:02d}_{agent_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"no CSV at {csv_path}")
    baseline, variant, personas = [], [], []
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    if "baseline" not in by_label:
        raise ValueError("no baseline rows in CSV")
    if variant_label not in by_label:
        raise ValueError(f"no rows for variant label '{variant_label}'. "
                          f"Available: {list(by_label.keys())}")
    base_rows = by_label["baseline"]
    var_rows = by_label[variant_label]
    if len(base_rows) != len(var_rows):
        raise ValueError(f"length mismatch: baseline={len(base_rows)} variant={len(var_rows)}")
    for b, v in zip(base_rows, var_rows):
        baseline.append(float(b["primary"]))
        variant.append(float(v["primary"]))
        personas.append(b["persona"])
    return baseline, variant, personas


def find_proposal_metadata(agent_id: str, iter_id: int, variant_idx: int) -> tuple[str, str] | None:
    """Return (prompt_text, rationale) from the candidate_preflight row that
    corresponds to this variant, or None if we can't find it.

    We match heuristically: latest candidate_preflight or rejected row for this
    agent that has adoption_data.iteration_id == iter_id.
    """
    with session_scope() as s:
        rows = s.execute(
            select(PromptVersion)
            .where(PromptVersion.agent_id == agent_id)
            .where(PromptVersion.status.in_(["candidate_preflight", "rejected"]))
            .order_by(PromptVersion.id.desc())
        ).scalars().all()
        for r in rows:
            ad = r.adoption_data or {}
            if ad.get("iteration_id") == iter_id:
                return r.prompt_text, ad.get("rationale") or r.rejection_reason or ""
    return None


def lift_to_disk(agent_id: str, version: int, text: str) -> Path:
    out_dir = REPO / "prompts" / agent_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"v{version:04d}.md"
    out_path.write_text(text)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, choices=["agent_1", "agent_2", "agent_3"])
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--variant", default="variant1",
                   help="CSV label of the variant to evaluate (e.g. 'variant0', 'variant1').")
    p.add_argument("--rationale-fallback",
                   default="retroactively adopted under multi-method consensus gate",
                   help="adoption_data.rationale to use if we can't find the original proposal text")
    p.add_argument("--dry-run", action="store_true",
                   help="show the decision without writing to DB or disk")
    args = p.parse_args()

    print(f"=== retroactive re-decide: {args.agent} iter {args.iter} {args.variant} ===\n")

    baseline, variant, personas = load_pair(args.agent, args.iter, args.variant)
    n = len(baseline)
    diff = sum(v - b for v, b in zip(variant, baseline)) / n
    print(f"loaded N={n} paired samples from CSV.")
    print(f"  baseline mean = {sum(baseline)/n:.4f}")
    print(f"  variant  mean = {sum(variant)/n:.4f}")
    print(f"  mean diff     = {diff:+.4f}\n")

    # ---- multi-method consensus
    mv = consensus(baseline, variant)
    ci = paired_bootstrap_ci(baseline, variant)
    cd = cohens_d(variant, baseline)

    print("multi-method decision surface:")
    print(f"  bootstrap 95% CI         = [{ci.ci_lower:+.4f}, {ci.ci_upper:+.4f}]  "
          f"({'lower > 0' if ci.ci_lower > 0 else 'lower NOT > 0'})")
    for r in mv.sig_results:
        sig = "SIG" if r.significant() else "ns"
        print(f"  {r.name:18s}  p={r.p_value:.4f}  {sig}  stat={r.statistic:+.3f}")
    for e in mv.effect_results:
        print(f"  {e.name:18s}  = {e.value:+.3f}  ({e.interpretation})")

    print()
    print(f"significance agreement: {mv.sig_votes_positive}/{mv.sig_total}")
    print(f"effect-size agreement : {mv.effect_votes_meaningful}/{mv.effect_total}")
    print(f"  → consensus.adopt() = {mv.adopt()}")
    print()

    if not mv.adopt():
        print("Not enough cross-method agreement to adopt. No changes written.")
        return 0

    print(">>> consensus says ADOPT. <<<\n")

    # ---- get the proposal text + rationale, if we can find them
    # variant1 etc. is the CSV label; the variant_idx in DB starts at 0
    variant_idx = int(args.variant.replace("variant", "")) if args.variant.startswith("variant") else 0
    meta = find_proposal_metadata(args.agent, args.iter, variant_idx)
    if meta is None:
        print(f"WARNING: could not locate the original proposal row in prompt_versions; "
              f"falling back to copying the current active prompt as the new version.")
        with session_scope() as s:
            from packages.storage.repos import get_active_prompt
            current = get_active_prompt(args.agent)
            prompt_text = current.prompt_text
            rationale = args.rationale_fallback
    else:
        prompt_text, rationale = meta

    # ---- decide the new version number (next available)
    with session_scope() as s:
        rows = s.execute(
            select(PromptVersion.version)
            .where(PromptVersion.agent_id == args.agent)
            .where(PromptVersion.version < 1000)   # ignore synthetic reject IDs (10000+)
        ).all()
        existing = {r[0] for r in rows}
    next_v = max(existing) + 1 if existing else 2

    print(f"  new version number: v{next_v}")
    print(f"  prompt tokens     : {len(prompt_text.split())} words / {len(prompt_text)} chars")
    print(f"  rationale         : {rationale[:200]}\n")

    if args.dry_run:
        print("(dry-run) — no DB or disk changes made.")
        return 0

    from packages.llm.token_guard import count_tokens
    new_id = upsert_prompt_version(
        agent_id=args.agent,
        version=next_v,
        prompt_text=prompt_text,
        prompt_tokens=count_tokens(prompt_text),
        status="active",
        adoption_data={
            "adopted_via": "retroactive_multi_method_consensus",
            "iteration_id": args.iter,
            "variant_label": args.variant,
            "n": n,
            "mean_diff": diff,
            "bootstrap_ci_lower": ci.ci_lower,
            "bootstrap_ci_upper": ci.ci_upper,
            "cohens_d": cd,
            "sig_votes_positive": mv.sig_votes_positive,
            "sig_total": mv.sig_total,
            "effect_votes_meaningful": mv.effect_votes_meaningful,
            "effect_total": mv.effect_total,
            "per_method": {
                "paired_t_test_p": next(r.p_value for r in mv.sig_results if r.name == "paired_t_test"),
                "wilcoxon_p":      next(r.p_value for r in mv.sig_results if r.name == "wilcoxon"),
                "permutation_p":   next(r.p_value for r in mv.sig_results if r.name == "permutation"),
                "hedges_g":        next(e.value   for e in mv.effect_results if e.name == "hedges_g"),
                "cliffs_delta":    next(e.value   for e in mv.effect_results if e.name == "cliffs_delta"),
            },
            "rationale": rationale,
        },
    )
    set_active_prompt(args.agent, new_id)
    print(f"wrote prompt_versions row id={new_id} status=active for {args.agent} v{next_v}")
    print(f"  active_prompt[{args.agent}] -> version_id={new_id}")

    out_path = lift_to_disk(args.agent, next_v, prompt_text)
    print(f"lifted to disk: {out_path}\n")

    print("Run `make report` to regenerate EVOLUTION_REPORT.md with the new adoption visible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
