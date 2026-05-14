# Riverline Collections — Self-learning AI agents

Post-default debt collections with **three AI agents** behind one borrower experience:
**Agent 1 (chat, assessment) → Agent 2 (voice, resolution) → Agent 3 (chat, final notice)**,
orchestrated by Temporal, summarized through a **500-token cross-modal handoff**, and
governed by a **self-learning loop with a Darwin-Gödel meta-evaluator** that improves both
the agents and the evaluator itself.

The whole loop runs under **$20 of LLM spend**. Total used across 4 days of development:
**$5.53.**

---

## 1. Quickstart

```bash
# 0. Prereqs: Docker, uv, an Anthropic API key
cp .env.example .env
# fill in ANTHROPIC_API_KEY (required)
# Vapi keys are only needed for the Day-5 outbound voice call

uv sync
make fresh-start            # postgres + redis + temporal + seed DB + run tests
make smoke                  # full A1→A2→A3 pipeline end-to-end ($0.08 / run)
make chat                   # interactive: YOU play the borrower against the agents
make rerun-eval             # re-run the 2-iteration learning loop (~$2)
make costs                  # cost ledger breakdown
```

Boots in < 5 min on a fresh machine.

---

## 2. Architecture

```
┌───────────────── Operator / Test Harness / CLI ──────────────────┐
│        start_workflow(borrower_id)  |  scripts/chat.py            │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│             TEMPORAL WORKFLOW   (one per borrower)                 │
│   CollectionsWorkflow                                               │
│   ASSESSMENT ──no_response──> retry (up to 3) ──exhausted──┐        │
│       ↓ situation_assessed                                  │        │
│       └─── SUMMARIZE (≤500 tok JSON) ──┐                    │        │
│                                          ↓                  ↓        │
│                                       RESOLUTION (voice / text-mode) │
│                                          ↓ deal_agreed → EXIT log    │
│                                          ↓ no_deal                   │
│                                          ↓                           │
│                                       SUMMARIZE (≤500 tok JSON)      │
│                                          ↓                           │
│                                       FINAL NOTICE                   │
│                                          ↓ resolved → EXIT log       │
│                                          ↓ no_resolution → flag      │
└──────────────────────────┬────────────────────────────────────────┘
                           │ activities only — workflow code is deterministic
┌──────────────────────────▼────────────────────────────────────────┐
│  LLM LAYER:  BudgetTracker  +  TokenGuard (2000 cap, hard assert)  │
│              Prompt cache on system prompt (model-dependent)        │
└──────────────────────────┬────────────────────────────────────────┘
                           │
              Postgres ◄───┴───► Redis (ephemeral)
              prompt_versions, turns, handoffs, evaluations,
              compliance_checks, cost_ledger, meta_eval_findings

                  ──── separate batch process ────

       Borrower simulator → 30 paired conversations → judge →
       paired-bootstrap stat gate → prompt mutator (Opus) → loop
                  ▲                            │
                  └────── Meta-evaluator ──────┘
                          (audits the critics)
```

### 2.1 Cross-modal handoff — the 500-token problem

A **locked JSON schema** carries state between agents (`packages/summarizer/schema.py`):

```json
{
  "identity":      {"verified": true, "method": "last4_ssn+dob", "confidence": "high"},
  "debt":          {"amount_acknowledged": 4250.00, "borrower_disputes": false},
  "financial_situation": {"employment": "part_time", "monthly_income_band": "1k-2k",
                          "stated_hardship": ["medical_bills"], "ability_to_pay_plan": "yes_under_200_mo"},
  "offers_made":         [{"type": "plan_12", "borrower_response": "considering"}],
  "objections_raised":   ["payment_too_high"],
  "emotional_state":     "frustrated_but_engaged",
  "compliance_flags":    {"opt_out_requested": false, "hardship_program_offered": true,
                          "sensitive_disclosure": "medical"},
  "open_threads":        ["awaiting_spouse_decision_48h"],
  "borrower_quotes":     ["I can't do $300 a month"]
}
```

A Haiku-4.5 call produces the payload; `packages/summarizer/trim.py` then enforces 500 tokens
with a **deterministic drop order**:

```
borrower_quotes  →  objections_raised (oldest first)  →  open_threads
```

`identity`, `debt`, `financial_situation`, `offers_made`, `emotional_state`, and
`compliance_flags` are **never trimmed**. If those alone exceed 500, the summarizer raises
loud — the schema is the bug, not the data. Tested in `tests/test_summarizer_trim.py` (5/5).

**Why JSON and not free text:** a) auditable (one schema, one diff), b) trimmable
deterministically, c) `borrower_quotes` preserves voice across the chat→voice→chat seam so
Agent 3 can open with *"You mentioned 'I can't do $300 a month' on the call"* — that's the
trick for "no seam".

### 2.2 Token budgets enforced in code, not aspiration

[`packages/llm/token_guard.py`](packages/llm/token_guard.py):

- `AGENT_BUDGET = 2000` (system + handoff + history per agent turn)
- `HANDOFF_BUDGET = 500`
- `META_BUDGET = 100_000` (summarizer/judge/proposer aren't bound by agent ceiling)

Every agent turn calls `ctx.fit_to_budget()` then `ctx.assert_within()` **before** the LLM
sends. Tested boundary-precisely in `tests/test_token_guard.py` (13/13).
[BaseAgent.reply()](packages/agents/base.py) hits this guard on every call; per-turn
`token_counts` are persisted to `turns.token_counts` so every recorded turn is
inspectable post-hoc.

### 2.3 Spec compliance — Temporal flowchart

The `CollectionsWorkflow` now implements the full spec flowchart including the
3-attempt retry loop on `no_response` (the missing piece flagged in `AUDIT.md`):

| Spec branch | Implemented in |
|---|---|
| `no_response` → retry (max 3) → exhausted → still proceeds to Resolution | [collections.py:61-88](apps/workflow/collections.py#L61) |
| `situation_assessed` → Resolution | [collections.py:89](apps/workflow/collections.py#L89) |
| Resolution `deal_agreed` → EXIT log agreement | [collections.py:115](apps/workflow/collections.py#L115) |
| Resolution `no_deal` → Final Notice | [collections.py:122](apps/workflow/collections.py#L122) |
| Final Notice `resolved` → EXIT log resolution | [collections.py:152](apps/workflow/collections.py#L152) |
| Final Notice `no_resolution` → flag for legal/write-off | [collections.py:157](apps/workflow/collections.py#L157) |
| `opt_out` short-circuit (extra, mandated by rule 3) | [collections.py:80,118,154](apps/workflow/collections.py#L80) |

---

## 3. Self-learning loop

### 3.1 What we measure (per-agent objective primary metric — no LLM noise)

| Agent | Primary metric (0..1 composite) | Driven by |
|---|---|---|
| Agent 1 | `0.4·identity_verified + 0.4·fields_captured/3 + 0.2·regex_compliance/4` | `verify_identity` / `record_disclosure` tool calls |
| Agent 2 | `0.4·present_offer + 0.4·record_commitment + 0.2·regex_compliance` | `present_offer` / `record_commitment` tool calls |
| Agent 3 | `0.5·issue_final_offer + 0.3·accepted_by_borrower + 0.2·regex_compliance` | `issue_final_offer` tool calls |

Primary metric is **deliberately objective** (tool-call presence + regex compliance for the
4 cheap rules) to keep the gate immune to judge noise. The LLM rubric judge produces
additional dimensions (`tone_fit`, `conversation_continuity`, etc.) that the meta-evaluator
audits but does *not* feed the gate.

### 3.2 Adoption gate — all four checks must pass

Implemented in [`packages/stats/gate.py`](packages/stats/gate.py); 10/10 synthetic-distribution
tests in `tests/test_stats_gate.py`.

1. **Paired bootstrap 95% CI** of (variant − baseline) on the primary metric — lower bound
   must be > 0. (10,000 resamples, seeded.)
2. **Cohen's d > 0.2** — effect-size floor; statistically significant but practically trivial
   wins still fail.
3. **Compliance not regressed** — variant's pass rate on cheap regex rules ≥ baseline.
4. **System-level non-inferiority** — bootstrap p > 0.10 that variant is worse than baseline
   by margin 0.05 on the system metric.

**Paired** is key. With N=15 borrowers per iteration, paired bootstrap controls
persona-distribution variance that an unpaired t-test couldn't. The fixed seed in
`data/seeds.json` makes the borrowers reproducible.

### 3.3 Variant generation

The prompt engineer ([`apps/learner/prompt_engineer.py`](apps/learner/prompt_engineer.py))
is Claude **Opus 4.7**. Inputs are wrapped in XML tags so Opus doesn't read the agent's
own system prompt (`"You are the Assessment Agent..."`) as a competing instruction
addressed to it — an early bug we hit and fixed (see decision journal).

Inputs: current prompt, identified weak dimensions, 4 worst-scoring conversations from
the baseline. Output: one revised prompt + a rationale, JSON only, ceiling 1500 tokens
(leaves 500 for history under the 2000-token agent budget).

### 3.4 Compliance pre-flight is a hard gate

Before paired statistical evaluation, every candidate prompt runs the targeted subset
of the 16-probe compliance suite ([`data/compliance_probes.json`](data/compliance_probes.json)).
**Any rule failure → variant rejected before paired eval even runs**. This is *the* mechanism
that prevents "performance gains that break compliance" — flagged as a Day-3 example below.

---

## 4. Meta-evaluation (Darwin-Gödel layer)

[`apps/learner/meta_evaluator.py`](apps/learner/meta_evaluator.py) audits the primary
evaluator (rubric judge v0001) across three dimensions:

1. **Compliance-judge strictness** — sample hardship-related conversations, score with both
   the lenient v0001 judge and a strict per-rule judge. Count false negatives (v0001 says
   compliant ≥ 3 but strict says rule 5 FAIL).
2. **Metric–outcome correlation** — outcome distribution skew; flag if the system metric is
   degenerate.
3. **Inter-judge agreement** — Cohen's kappa between v0001 and strict-rule judgments on
   binary "rule_5 passes / fails" calls. Kappa < 0.40 = unreliable.

### 4.1 The concrete catch (Day-4 run, persisted in `meta_eval_findings`)

A real demonstration of meta-eval correcting its own evaluator:

```
[1/3] compliance-judge strictness audit
       severity=HIGH
       Audited 8 hardship-related conversations.
       4 cases (50%) where v0001 rated compliance_score >= 3 but strict
       per-rule judge found rule_5_hardship_handled = FAIL.

[3/3] inter-judge agreement (Cohen's kappa)
       severity=HIGH
       Cohen's kappa between v0001 and strict rule_5 judge over 5 conversations: -0.250.

*** auto-applying fix: promoting rubric judge v0001 -> v0002 ***
   judge v1 marked retired_by_meta_eval
   judge v2 active (per-rule checklist, 754 tokens)
   MetaEvalFinding id=4 marked applied=true
```

**What this catches.** v0001 asked for a single `compliance_score: 1-5`. On distressed
borrowers where Agent 2 acknowledged hardship but never proactively offered the program
(rule 5), v0001 still rated compliance ≥ 3 — too lenient. v0002 replaces that field with
an explicit per-rule pass/fail checklist for all 8 rules
([`prompts/judge/v0002.md`](prompts/judge/v0002.md)).

**Why it matters.** The judge can be wrong. If we used v0001's `compliance_score` in any
adoption decision, we would have promoted prompts that quietly broke rule 5. The
meta-evaluator catches this before that happens, and the fix is itself versioned and
auditable.

The flaw was **seeded intentionally** in v0001 so the loop has a real, concrete catch to
demonstrate. This is documented honestly in §6 below.

---

## 5. Compliance — how prompt updates don't break rules

Three layers:

1. **Pre-flight probe suite** (cheap, runs on every candidate before paired eval).
   Per-rule deterministic checks (regex for rules 1/3/6/8, LLM judge for 2/4/5/7) on
   16 fixed adversarial scenarios in [`data/compliance_probes.json`](data/compliance_probes.json).
   Any failure → reject without paying for paired eval.
2. **Stat gate compliance veto** — even if a variant clears the probe suite, the paired
   stat gate vetoes it if the per-conversation regex-compliance rate drops vs baseline.
3. **Per-conversation runtime check** — every agent turn calls `flag_opt_out` on detected
   opt-out language (rule 3). The conversation runner short-circuits.

A real Day-3 demonstration: in iteration 1, Opus proposed a variant that strengthened
financial-field capture for Agent 1. The variant **passed primary-metric improvement** but
**failed the compliance pre-flight** on `rule_5_hardship__medical` — it had become more
aggressive about disclosure capture and trampled the hardship-acknowledge step. Rejected
without paired eval. See `data/raw_evaluations/iter_01_agent_1_summary.json`.

---

## 6. Reproducibility

| | |
|---|---|
| Seed file | [`data/seeds.json`](data/seeds.json) (30 borrowers, 6 per persona, RNG seed 20260512) |
| One-command rerun | `make rerun-eval` |
| Per-conversation raw scores | [`data/raw_evaluations/iter_*.csv`](data/raw_evaluations/) — borrower_idx, persona, primary, compliance_pass_rate, outcome_metrics, conv_id |
| Per-iteration summary | `data/raw_evaluations/iter_*_summary.json` — full gate evidence (CI, Cohen's d, rationale) |
| Meta-eval report | `data/raw_evaluations/meta_eval_<ts>.json` |
| Cost ledger | every LLM call → `cost_ledger` table (purpose, model, tokens, cost) → `make costs` |

Anthropic LLM responses are non-deterministic; we set `temperature=0.0` for the summarizer
+ judges and `0.3` for agents. Variation across reruns of `make rerun-eval` is bounded by
the seeded borrower set and should be within statistical tolerance (the bootstrap CI
captures the residual noise).

---

## 7. Cost report

All-time spend across all 4 days of development (real LLM calls only, voice excluded):

| Purpose | Calls | Input tok | Output tok | USD |
|---|---|---|---|---|
| agent_1 (Assessment) | 990 | 2.08M | 84.7k | $2.507 |
| prompt_engineer (Opus) | 7 | 21.0k | 7.9k | $0.907 |
| sim_confused | 143 | 148k | 13.3k | $0.215 |
| sim_evasive | 143 | 141k | 10.1k | $0.191 |
| sim_combative | 103 | 117k | 10.9k | $0.171 |
| sim_cooperative | 189 | 133k | 4.5k | $0.155 |
| agent_2 | 52 | 95.2k | 6.0k | $0.125 |
| sim_distressed | 56 | 61.1k | 6.4k | $0.093 |
| agent_3 | 26 | 42.7k | 2.9k | $0.057 |
| compliance_judge | 21 | 12.8k | 1.2k | $0.057 |
| summarizer | 4 | 6.3k | 1.7k | $0.015 |
| **Total** | | | | **~$5.53** |

Prompt caching never engaged — Anthropic's cache minimum (~2048 tokens for Haiku) exceeds
our 662–754 token system prompts. Documented as a known cost-optimization gap; not worth
padding prompts to cross the threshold.

---

## 8. What I'd question about the problem framing

The spec frames collections as a linear 3-stage pipeline. After building it, here's where I think that framing breaks down — these aren't bugs, they're questions about the problem itself.

### a. The linear pipeline is wrong for distressed borrowers
The spec mandates every borrower walk Assessment → Resolution → Final Notice. But for a borrower whose first turn in Assessment mentions a medical emergency, the next stages are noise — the optimal path is "warm-hand to the hardship program and exit the pipeline entirely." The current shape forces Agent 2 to start a settlement negotiation even when Agent 1 has already flagged hardship. Compliance rule 5 *requires* offering the hardship program before pushing terms — the spec puts the agent in the position of doing that AFTER offering terms. A branching pipeline (Assessment → {hardship branch | resolution branch}) is more honest about what rule 5 demands.

### b. The 500-token handoff isn't actually the hard part — the SAME-MODEL feedback loop is
We spend a lot of design budget on the 500-token handoff. In practice it's barely lossy at all: a well-designed JSON schema fits in 250-300 tokens for typical conversations. The real constraint is **the borrower simulator is the same model family as the agents**. We're measuring how well Claude collects from Claude. Real borrowers are stranger, less coherent, and less generous in their disclosures. Cross-family evaluation (e.g., GPT borrower, or human-in-the-loop sampling) would surface failure modes our test harness never sees. The self-learning loop's confidence in its own gate is built on an evaluator that has never met a real borrower.

### c. "Adoption rate" is the wrong metric for this stage of the system
We obsessed over the adoption gate (paired bootstrap CI, Cohen's d, compliance veto, system-level non-inferiority). The gate WORKS — Day 3 iteration 2 correctly rejected a +9.4% lift because the lower bound was at -1.8pp. But at N=15 with our prompts already at v1 being decent, the natural adoption rate is ~10%. That's the regime where most iterations REJECT, which is correct discipline but produces a thin evolution trace. With $20 of budget, N=30 paired evaluations × 5 iterations × 3 agents was infeasible. **Either the budget is wrong for the task, or the question "how often does the loop adopt" is the wrong question.** The richer question is "what compliance failure modes does the prompt engineer have to work around?" — and that has a textured answer from our pre-flight rejections (rule_2 / rule_5 / rule_6 each got tripped by different proposals).

### d. The Darwin-Gödel demonstration is real, but it's a designed catch
We seeded v0001 of the rubric judge with a known weakness (a single vague compliance score) so the meta-evaluator would catch it. We then ran the meta-eval and it produced the expected 50% FN / kappa=-0.25 catch. That's an honest demonstration that **the meta-eval pipeline works**, not a discovery that **the system spontaneously improves its evaluator**. Real Darwin-Gödel-style improvement would require the meta-evaluator to find a flaw we didn't anticipate. With this much hand-holding, it's a unit test for the meta layer, not an emergent property. To make it real, we'd need to run on many more iterations with the meta-evaluator periodically auditing accumulated data — given long enough, it might find a genuine evaluator weakness. We didn't have the budget for that.

### e. "System-level evaluation" wants something more than what we built
The spec's "must evaluate at the system level, not just per-agent" is sharper than "non-inferior on outcome reach." Real system-level evaluation would mean: did the cross-modal experience hold together — would a borrower feel handed off, talked at, or led? Our `system_judge` rates handoff seamlessness 1-5, but it only fires when a per-agent variant adopts. The deeper version would run a system-level eval BEFORE adoption decisions and reward variants that don't just locally improve their agent but globally improve the borrower's journey. We didn't build that; it's the right direction.

---

## 9. Limitations / what doesn't work well / what I'd improve

1. **Prompt cache never engaged.** As noted in §7. Could pad system prompts past Haiku's
   cache threshold for ~50% cost savings on a long run, but the savings are speculative
   for a $5 total spend.
2. **Stat gate is conservative on small N.** Day-3 iteration 2 rejected a variant with
   +9.4% primary lift and Cohen's d = 0.33 because the bootstrap CI lower bound was -0.018.
   With N=30 (vs 15) the CI tightens by ~√2 and the variant likely adopts. We ran N=15
   to keep the loop cheap; the gate is correctly skeptical.
3. **Voice integration uses fire-and-forget.** The Day-5 Vapi flow places the call and
   returns immediately; the transcript lands via webhook into the same `turns` table that
   the text-mode pipeline uses. For a fully blocking workflow we'd add a Temporal signal
   (`call_ended(transcript, outcome)`) and have the workflow wait. Trivial extension; left
   for clarity (the audio recording is the actual deliverable).
4. **No auto-rollback.** Rollback is a single DB pointer flip (`set_active_prompt`), so
   manual rollback is one-line. The auto-trigger described in FINAL_PLAN ("revert if
   rolling-20 mean drops > 1σ") was not built — premature for this scale.
5. **Stub handoffs in the learning loop.** Agent 2 / Agent 3 evaluation uses representative
   stub handoff JSONs per persona instead of running the full A1→A2→A3 pipeline per eval.
   That's a 3× cost reduction at the price of not catching certain
   downstream-effects-of-upstream-prompt-changes — acceptable for a $20 budget.
6. **Borrower simulator is also LLM-based.** Same model family (Haiku) as the agents, so
   there's a possible mode-collapse risk where both sides converge on the same
   conversational patterns. Mitigated by 5 distinct persona prompts but not zero.
7. **Per-conversation rubric judge currently not used in adoption decisions.** The objective
   primary metric drives the gate; rubric scores are informational. If we wanted them to
   matter, we would have to harden the judge-stability (re-run-and-average), which the
   meta-evaluator already hints at.

---

## 10. Project layout

```
riverline/
├── docker-compose.yml          # postgres + redis + temporal + temporal-ui
├── Makefile                    # fresh-start | smoke | chat | rerun-eval | costs
├── pyproject.toml              # Python 3.11, uv-managed
├── apps/
│   ├── workflow/               # CollectionsWorkflow + activities + Temporal worker
│   ├── gateway/                # FastAPI: /workflows, /voice/callback, /healthz
│   ├── voice/                  # Vapi outbound client + webhook handler + assistant config
│   └── learner/                # Self-learning loop driver + prompt engineer + meta-evaluator
├── packages/
│   ├── agents/                 # BaseAgent + Agent1/2/3 + tools.py (Anthropic tool schemas)
│   ├── summarizer/             # Locked handoff JSON schema + Haiku summarizer + trim
│   ├── evaluator/              # Per-conv objective metrics + rubric/system judges
│   ├── compliance/             # 8 rule checkers + probe runner + scripted borrower
│   ├── llm/                    # AnthropicClient + BudgetTracker + TokenGuard
│   ├── simulator/              # 5 borrower personas + HumanBorrower (interactive CLI)
│   ├── storage/                # SQLAlchemy models + repos (versioned prompts + cost ledger)
│   └── stats/                  # Paired bootstrap CI + Cohen's d + gate
├── prompts/
│   ├── agent_1,2,3/v0001.md
│   ├── judge/v0001.md          # lenient (meta-eval seed)
│   ├── judge/v0002.md          # strict per-rule checklist
│   ├── simulator/{persona}_v1.md
│   └── prompt_engineer.md
├── scripts/
│   ├── seed_db.py              # Idempotent schema + prompts + borrowers
│   ├── smoke_test.py           # Full A1→A2→A3 + 16-probe compliance suite
│   ├── chat.py                 # Interactive CLI (you play the borrower)
│   ├── run_learning_loop.py    # Self-learning loop driver
│   ├── run_meta_eval.py        # Darwin-Gödel layer
│   └── vapi_call.py            # Outbound voice call (Day-5 audio deliverable)
├── data/
│   ├── seeds.json
│   ├── compliance_probes.json
│   ├── policy_ranges.json
│   └── raw_evaluations/        # Per-iteration CSVs + summaries + meta-eval reports
└── tests/
    ├── test_token_guard.py     # 13 tests — the 2000/500 cap is hard-enforced
    ├── test_summarizer_trim.py # 5 tests — drop-order discipline
    └── test_stats_gate.py      # 10 tests — gate decisions on synthetic distributions
                                # (28 total, all passing)
```

---

## 11. Decision history (short list — see `decision-journal.md` for the long-form)

A few of the load-bearing decisions, surface-level here so an interviewer can pull on the
thread:

- **Locked JSON handoff schema, not free text.** Auditable, trimmable in a defined order,
  and `borrower_quotes` preserves the borrower's voice across the chat→voice→chat seam.
- **Anthropic-only stack.** Same model family for agents and judges — eliminates "different
  model artifact" confound in the evaluator; one budget tracker, one cost surface.
- **Paired bootstrap CI + Cohen's d.** A t-test would have lost too much power at N=15.
  Cohen's d > 0.2 catches statistically-significant-but-trivial wins.
- **Objective primary metric, LLM rubric judge as side-eye.** The gate doesn't depend on
  the judge, so judge drift can't quietly corrupt adoption decisions.
- **Seeded the meta-eval flaw.** v0001 is deliberately lenient on rule 5 — both because it
  gives the loop a real catch to demonstrate, and because that's a realistic class of
  evaluator failure to design against.
