# Riverline Collections

Post-default debt collections with three AI agents (chat → voice → chat) behind one borrower experience. Temporal-orchestrated, with a 500-token cross-modal handoff and a self-learning loop that improves both the agents and the evaluator.

Total LLM spend across all development: **about $10 of $20**.

---

## Quickstart

```bash
# Prereqs: Docker, uv, an Anthropic API key
cp .env.example .env             # then fill in ANTHROPIC_API_KEY
uv sync
make fresh-start                 # postgres + redis + temporal + seed + tests
make smoke                       # full A1 → A2 → A3 end-to-end (~$0.08)
make chat                        # you play the borrower in the terminal
make rerun-eval                  # re-run the learning loop (~$2)
make meta-eval                   # run the Darwin-Gödel layer
make costs                       # cost ledger breakdown
```

Cold boot to working pipeline: under 5 minutes on a fresh machine.

---

## Architecture

```
Operator / CLI
    │
    ▼
Temporal workflow (one per borrower)
    │
    ASSESSMENT  ─ no_response ─►  retry (max 3)  ─ exhausted ─┐
       │ situation_assessed                                     │
       ▼                                                        │
    summarize  ≤500-token JSON ─────────────────────────────┐   │
                                                              ▼   ▼
                                                       RESOLUTION (voice or text-mode)
                                                              │ deal_agreed → EXIT log
                                                              │ no_deal
                                                              ▼
                                                       summarize  ≤500-token JSON
                                                              │
                                                              ▼
                                                       FINAL NOTICE
                                                              │ resolved → EXIT
                                                              │ no_resolution → flag for legal / writeoff

Activities (all LLM calls live here):
   run_chat_agent, run_voice_agent, summarize_handoff, log_agreement, flag_for_legal_or_writeoff

Storage: Postgres for prompts, conversations, handoffs, evaluations, compliance_checks,
cost_ledger, meta_eval_findings. Redis for ephemeral session state.

Separate batch process:
   borrower simulator → 30 paired conversations → judge → paired-bootstrap stat gate
   → prompt mutator (Opus) → adopt or reject
                            ▲
                            │
                  meta-evaluator (audits the critics)
```

The workflow code is deterministic. Every LLM call, clock read, and random draw happens in activities. The token budget is hard-asserted before each agent call.

### Cross-modal handoff (500 tokens)

A locked JSON schema in `packages/summarizer/schema.py`:

```json
{
  "identity":      {"verified": true, "method": "last4_ssn", "confidence": "high"},
  "debt":          {"amount_acknowledged": 4250.00, "borrower_disputes": false},
  "financial_situation": {
    "employment": "part_time",
    "monthly_income_band": "1k-2k",
    "stated_hardship": ["medical_bills"],
    "ability_to_pay_plan": "yes_under_200_mo"
  },
  "offers_made":         [],
  "objections_raised":   ["payment_too_high"],
  "emotional_state":     "frustrated_but_engaged",
  "compliance_flags": {
    "opt_out_requested": false,
    "hardship_program_offered": true,
    "sensitive_disclosure": "medical"
  },
  "open_threads":        ["awaiting_spouse_decision_48h"],
  "borrower_quotes":     ["I can't do $300 a month"]
}
```

A Haiku call produces the payload. `packages/summarizer/trim.py` then trims deterministically when over 500 tokens:

```
borrower_quotes  →  objections_raised (oldest first)  →  open_threads
```

`identity`, `debt`, `financial_situation`, `offers_made`, `emotional_state`, and `compliance_flags` are never trimmed. If they alone exceed 500 the summarizer raises loud. Five tests in `tests/test_summarizer_trim.py` cover the drop order.

The `borrower_quotes` field is the part that makes the chat → voice → chat seam disappear: Agent 3 can open with "you mentioned 'I can't do $300 a month' on the call" and the borrower feels continuity.

### Token budgets enforced in code

`packages/llm/token_guard.py`:

```
AGENT_BUDGET   = 2000     (system + handoff + history per agent turn)
HANDOFF_BUDGET = 500
META_BUDGET    = 100_000  (summarizer / judges / proposer)
```

Every agent turn calls `ctx.fit_to_budget()` and `ctx.assert_within()` before the LLM. The assertions throw if the budget is exceeded. Per-turn `token_counts` are persisted to the `turns` table so every recorded turn is inspectable. Boundary tests in `tests/test_token_guard.py` cover the exact 2000-token edge.

### Spec compliance: the Temporal flowchart

```
no_response → retry up to 3 attempts → exhausted → still proceed to Resolution
```

Implemented at `apps/workflow/collections.py:61-88`. The `opt_out` short-circuit at every stage is a compliance-rule-3 addition not in the spec flowchart but mandated by the rules.

---

## Self-learning loop

### What I measure (per-conversation primary metric, no LLM)

Tool-call-driven composite, in [0..1]:

```
Agent 1 = 0.4 * identity_verified            + 0.4 * fields_captured/3   + 0.2 * regex_compliance/4
Agent 2 = 0.4 * present_offer                + 0.4 * record_commitment   + 0.2 * regex_compliance/4
Agent 3 = 0.5 * issue_final_offer            + 0.3 * accepted_by_borrower + 0.2 * regex_compliance/4
```

The primary metric is deliberately objective. The LLM rubric judge produces additional dimensions (tone, continuity, objection handling, task completion), but those do not feed the adoption gate. The gate is therefore immune to judge noise.

### Adoption gate (all four must hold)

`packages/stats/gate.py` — 10/10 synthetic-distribution tests pass.

1. **Paired bootstrap 95% CI** of (variant − baseline) on the primary metric — lower bound > 0. 10,000 resamples, seeded.
2. **Cohen's d > 0.1** — small-effect floor. Catches significant-but-trivial wins.
3. **Compliance not regressed** — variant's pass rate on the cheap regex rules ≥ baseline.
4. **System non-inferiority** — bootstrap p > 0.10 that the variant is worse than baseline by margin 0.05 on the system outcome metric.

After a variant clears the per-agent gate (for Agent 1 and Agent 2), the loop also runs `system_judge` on N=3 full A1→A2→A3 pipelines and rejects the variant if handoff seamlessness drops more than 0.5 on the 1-5 scale.

Paired evaluation is the point. With N=15 the persona variance is large; pairing the variant against the same borrowers as the baseline kills most of it. Bootstrap because the primary metric isn't normally distributed.

### Variant generation

Claude Opus 4.7 proposes prompt variants in `apps/learner/prompt_engineer.py`. Inputs are wrapped in XML tags because Opus reads markdown headers inside `<current_prompt>` blocks as instructions addressed to itself.

Inputs: current prompt, weak dimensions identified from the baseline scores, four worst-scoring conversations. Output: one revised prompt + a rationale, JSON only, ceiling 1500 tokens (leaves 500 for history under the 2000-token agent budget).

### Compliance pre-flight is a hard gate

Every candidate prompt runs the 16-probe compliance suite (`data/compliance_probes.json`) targeting the agent under test, before any paired statistical evaluation. Any rule failure rejects the variant without paying for paired eval. This is the mechanism that prevents "performance gains that break compliance."

### Baseline reuse

Before running fresh baseline conversations, the loop checks for stored conversations on the same active prompt version, agent_id, and balanced personas. If enough exist, it reuses them and pairs the variant against the same borrowers. Cuts iteration cost roughly in half.

---

## Meta-evaluation (Darwin-Gödel)

`apps/learner/meta_evaluator.py` runs three audits:

1. **Compliance-judge strictness** — sample hardship-related conversations, score with the lenient rubric judge and the strict per-rule judge. Count false negatives (rubric says compliant, strict says rule 5 failed).
2. **Metric-outcome correlation** — outcome-distribution skew on recent learning-loop conversations.
3. **Inter-judge agreement** — Cohen's kappa between the lenient judge and the strict rule_5 judge. Kappa < 0.40 is unreliable.

When audit 1 or 3 returns severity=high, the loop auto-promotes the rubric judge from v0001 to v0002 (per-rule checklist). The retired version is marked `retired_by_meta_eval` and the meta-finding row gets `applied=true`.

### Concrete catch (logged in `meta_eval_findings`)

```
Audit 1: 8 hardship conversations audited.
         4 (50%) cases where v0001 rated compliance_score >= 3
         but strict per-rule judge said rule_5_hardship_handled = FAIL.
         severity = HIGH

Audit 3: Cohen's kappa between v0001 and strict rule_5 = -0.25
         (negative = systematic disagreement, worse than chance)
         severity = HIGH

Action: promoted rubric judge v0001 -> v0002 (per-rule checklist)
        judge v1 status = retired_by_meta_eval
        meta_eval_findings.id = 7, applied = true
```

The v0001 weakness was deliberate — a single vague compliance score with no per-rule breakdown. The meta-evaluator was supposed to find that. The point of the demonstration is that the audit pipeline works, not that the system spontaneously discovered an evaluator flaw.

---

## Compliance

Three layers:

1. **Pre-flight probe suite** runs on every candidate before paired eval. 16 fixed adversarial scenarios in `data/compliance_probes.json`, covering all 8 rules. Regex checks for rules 1, 3, 6, 8. Sonnet judge for rules 2, 4, 5, 7. Any probe failure rejects the variant without further evaluation.
2. **Stat gate compliance veto** — even if a variant clears the probe suite, the gate vetoes it if the per-conversation regex compliance rate drops below baseline.
3. **Per-conversation runtime check** — agents call `flag_opt_out` on detected opt-out language (rule 3). The conversation runner short-circuits.

Example from a Day 3 iteration: Opus proposed a variant that strengthened Agent 1's financial-field capture. The variant improved the primary metric. The pre-flight rejected it because it had become more aggressive about disclosure capture and trampled rule 5 (hardship handling). The audit row is in `prompt_versions`.

---

## Reproducibility

| | |
|---|---|
| Seed file | `data/seeds.json` (30 borrowers, 6 per persona, RNG seed 20260512) |
| One-command rerun | `make rerun-eval` |
| Per-conversation raw scores | `data/raw_evaluations/iter_*.csv` — borrower index, persona, primary, compliance_pass_rate, outcome_metrics, conv_id |
| Per-iteration summary | `data/raw_evaluations/iter_*_summary.json` — full gate evidence (CI, Cohen's d, rationale) |
| Meta-eval report | `data/raw_evaluations/meta_eval_<ts>.json` |
| Cost ledger | every LLM call logged to `cost_ledger`; `make costs` aggregates per purpose |

Anthropic responses are not deterministic. The summarizer and judges run at `temperature=0.0`; agents and simulator at `0.3`. Reruns of `make rerun-eval` should reproduce primary metric means to within ±5pp and Cohen's d to within ±0.10 of reported numbers. The bootstrap CI captures the residual variance.

---

## Cost report

Across all development to date:

| Purpose | Calls | Input tok | Output tok | USD |
|---|---|---|---|---|
| agent_1 (Assessment) | ~1.2k | 2.7M | 110k | $3.1 |
| prompt_engineer (Opus) | ~15 | 60k | 25k | $1.8 |
| compliance_judge / rubric_judge (Sonnet) | ~70 | 100k | 8k | $0.6 |
| sim_* (5 personas, Haiku) | ~700 | 750k | 60k | $1.0 |
| agent_2 / agent_3 | ~150 | 200k | 18k | $0.4 |
| summarizer | ~20 | 30k | 10k | $0.1 |
| **Total** | | | | **~$10** |

Prompt caching never engaged. Haiku's cache minimum is ~2048 tokens; the agent system prompts are 660-900 tokens. Documented as a known optimization gap.

---

## Limitations

1. **Stat gate is conservative on small N.** A Day 3 iteration rejected a variant with +9.4% primary lift and Cohen's d = 0.33 because the bootstrap CI lower bound was at -0.018. With N=30 the CI tightens by about √2 and the variant adopts. N=15 was the budget-constrained choice.
2. **Voice integration is fire-and-forget.** The Vapi flow places the call and returns immediately; transcripts land via webhook into the same `turns` table the text-mode pipeline uses. A fully blocking workflow would use a Temporal `call_ended` signal.
3. **No auto-rollback trigger.** Manual rollback is a one-row SQL update on `active_prompt`. The "revert if rolling-20 mean drops > 1σ" trigger described in early planning was not built.
4. **Stub handoffs in the learning loop.** Agent 2 and Agent 3 evaluation use representative stub handoff JSONs per persona instead of running A1→A2→A3 for every paired evaluation. The system-level check (handoff seamlessness on N=3 full pipelines) partly compensates, but only fires post-per-agent-gate.
5. **Borrower simulator and agents are same model family.** Risk of mode collapse where both sides converge on the same conversational patterns. Mitigated by 5 distinct persona prompts; not eliminated.
6. **Rubric judge is not in the adoption decision.** Only the objective primary metric drives the gate. If the rubric judge fed adoption, judge-noise variance would need to be reduced first (re-run-and-average), which the meta-evaluator already hints at.
7. **Prompt cache never engaged.** Haiku's cache minimum is bigger than the system prompts. Padding past the threshold for cache savings would be wasteful at $10 total spend.

---

## Project layout

```
riverline/
├── docker-compose.yml          postgres + redis + temporal + temporal-ui
├── Makefile                    fresh-start | smoke | chat | rerun-eval | meta-eval | lift-prompts | costs
├── pyproject.toml              Python 3.11, uv-managed
├── apps/
│   ├── workflow/               Temporal workflow + activities + worker
│   ├── gateway/                FastAPI: /workflows, /voice/callback, /healthz
│   ├── voice/                  Vapi outbound client + webhook + assistant config
│   └── learner/                learning loop + prompt engineer + meta-evaluator
├── packages/
│   ├── agents/                 BaseAgent + Agent1/2/3 + tools.py (Anthropic tool schemas)
│   ├── summarizer/             locked handoff JSON schema + Haiku summarizer + trim
│   ├── evaluator/              objective metrics + rubric judge + system judge
│   ├── compliance/             8 rule checkers + probe runner + scripted borrower
│   ├── llm/                    AnthropicClient + BudgetTracker + TokenGuard
│   ├── simulator/              5 borrower personas + HumanBorrower (interactive)
│   ├── storage/                SQLAlchemy models + repos (versioned prompts + cost ledger)
│   └── stats/                  paired bootstrap CI + Cohen's d + gate
├── prompts/
│   ├── agent_1,2,3/v0001.md    initial prompts
│   ├── judge/v0001.md          lenient rubric (meta-eval seed)
│   ├── judge/v0002.md          strict per-rule checklist (after meta-eval swap)
│   ├── simulator/{persona}_v1.md
│   └── prompt_engineer.md
├── scripts/
│   ├── seed_db.py              idempotent schema + prompts + borrowers
│   ├── smoke_test.py           full A1→A2→A3 + 16-probe compliance suite
│   ├── chat.py                 interactive CLI (you play the borrower)
│   ├── run_learning_loop.py    self-learning loop driver
│   ├── run_meta_eval.py        Darwin-Gödel layer
│   ├── lift_active_prompts.py  export currently-active prompts back to disk
│   └── vapi_call.py            outbound voice call for the audio deliverable
├── data/
│   ├── seeds.json
│   ├── compliance_probes.json
│   ├── policy_ranges.json
│   └── raw_evaluations/        per-iteration CSVs + summaries + meta-eval reports
└── tests/
    ├── test_token_guard.py     13 tests: 2000/500 cap is hard-enforced
    ├── test_summarizer_trim.py  5 tests: drop-order discipline
    └── test_stats_gate.py      10 tests: gate decisions on synthetic distributions
                                28 total, all passing
```
