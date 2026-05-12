# Riverline Collections — FINAL Implementation Plan

> **Status:** decisions locked, ready to build. This supersedes `PLAN.md` (my v1) and `sonnet.md` (Sonnet's draft). Both source plans kept in the repo for reference; this file is the contract.

---

## 0. What changed from v1 (and why)

Merged the best of both drafts plus the architecture diagram. Concretely:

| Topic | v1 (mine) | sonnet.md | **FINAL** (this doc) | Why |
|---|---|---|---|---|
| LLM stack | Anthropic only | Claude (agents) + GPT-4o-mini (sim/eval) | **Anthropic only** (Haiku/Sonnet/Opus) | Prompt caching across iterations is the main cost lever; mixing providers complicates the budget tracker and confounds eval. |
| Stats gate | Paired bootstrap CI | t-test + Cohen's d | **Paired bootstrap CI + Cohen's d + compliance gate** | Paired design > unpaired on N=30. Cohen's d adds an effect-size floor that catches statistically-significant-but-trivial wins. |
| Handoff format | Structured JSON w/ schema | "summarizer Haiku call, free text under instructions" | **Structured JSON (locked schema, enforced)** | Free text drifts under prompt evolution. Schema is auditable and the budget-trimming logic has a deterministic drop order. |
| Compliance gate | 20-conv probe suite (per-rule) | 10 adversarial scenarios | **20 probes (8 rules × ≥2 trigger types) + LLM judge for nuanced rules** | Per-rule coverage with named triggers — auditable and reproducible. |
| Meta-eval demo | 3 candidates (lenient judge / metric-outcome decoupling / persona drift) | Hardship sub-check gap | **Seed the lenient-compliance-judge flaw (rule 5 hardship); back it up with a metric-correlation check** | Specific, demonstrable, and a single coherent story for the writeup/interview. |
| Voice strategy | Text-mode for eval + 1 real call for recording | Vapi end-to-end | **Vapi end-to-end; text-mode is the fallback if Vapi blocks** | Spec wants a real recording. Vapi is simple enough to get one call working. Text-mode shim stays for cheap iteration on Agent 2's prompt during learning loop. |
| Rollback rule | DB pointer flip | Auto-rollback if live mean drops >1σ over 20 convs | **DB pointer flip + auto-rollback trigger** | Both. Manual rollback always available; auto-rollback catches drift. |

---

## 1. Locked decisions (kick this off — no more debate)

1. **Language:** Python 3.11.
2. **Orchestration:** Temporal (Python SDK, `temporalio`).
3. **LLM provider:** Anthropic.
   - Agents (1, 2, 3) → **Claude Haiku 4.5**
   - Borrower simulator → **Claude Haiku 4.5**
   - LLM-as-judge (rubric + system-level + compliance) → **Claude Sonnet 4.6**
   - Prompt Engineer (proposer) + Meta-Evaluator → **Claude Opus 4.7**
4. **Voice:** Vapi (outbound). Webhook posts transcript on call end. Public webhook URL via Cloudflare Tunnel.
5. **Storage:** Postgres 16 (prompts, evals, audit, cost). Redis 7 (live chat session state, handoff payloads).
6. **API:** FastAPI.
7. **Tokenizer:** `tiktoken` (cl100k_base). Documented as a strict overcount vs Claude → our 2000-token cap is *stricter* than spec.
8. **Stats:** `scipy` + `numpy`. Paired bootstrap CI (10k resamples) + Cohen's d.
9. **N per iteration:** **30 borrowers**, 6 per persona, fixed seed.
10. **Adoption gate (all must hold):**
    - Paired bootstrap 95% CI of (variant − incumbent) on primary metric: lower bound > 0
    - Cohen's d > 0.2 (small-but-real effect)
    - Compliance pass rate ≥ incumbent (zero regression tolerance)
    - System-level resolution rate not worse with p > 0.10 (non-inferiority)
11. **Auto-rollback:** if active version's rolling-20-conversation mean on primary metric drops > 1σ below the prior version's mean → revert pointer + alert.
12. **Cost ceiling:** $20 hard. Kill switch at $18. Budget tracker wraps every LLM call.
13. **Container:** Docker Compose, single `docker compose up` boots everything in <5 min (pre-pulled images, Temporal dev container).
14. **Deliverable contract:** seeds.json + `make rerun-eval` regenerates evolution report deterministically (±tolerance noted in writeup).

---

## 2. System architecture (one diagram, four layers)

```
┌────────────────────────── OPERATOR / TEST HARNESS ───────────────────────────┐
│   CLI: start_workflow(borrower_id)    |    make rerun-eval                   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│                          TEMPORAL WORKFLOW (per borrower)                     │
│   CollectionsWorkflow                                                         │
│   ├ ASSESSMENT  ─▶  SUMMARIZE(≤500)  ─▶  RESOLUTION  ─▶  SUMMARIZE(≤500)      │
│   │                                                       │                   │
│   │                                                       ▼                   │
│   │                                                  FINAL NOTICE             │
│   │                                                       │                   │
│   └───────────────────────  OUTCOME → audit log  ◀────────┘                   │
│   Signals: borrower_replied, call_ended, opt_out_requested                    │
└────────┬────────────────────────┬───────────────────────┬─────────────────────┘
         │                        │                       │
   ChatActivity              VoiceActivity         SummarizerActivity
         │                        │                       │
         └────────────────────────┴───────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────────────┐
│   LLM LAYER:   BudgetTracker  +  TokenGuard (2000 ceiling, hard assert)       │
│                 Prompt cache (system prompt cached across iterations)         │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────────────┐
│   STORAGE                                                                     │
│   Postgres: prompt_versions, turns, handoffs, evals, cost_ledger,             │
│             meta_eval_findings, compliance_checks                             │
│   Redis:    session:{borrower_id}, handoff:{borrower_id}                      │
└──────────────────────────────────────────────────────────────────────────────┘

                         (independent batch process)
┌──────────────────────────────────────────────────────────────────────────────┐
│   SELF-LEARNING LOOP                                                          │
│   simulator → 30 conversations → judge → stats gate → prompt mutator          │
│                                              ▲                                │
│                                              │                                │
│                                       META-EVALUATOR                          │
│                                  (critiques the critics)                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

This mirrors the diagram in the screenshot. Components 1:1.

---

## 3. Token-budget enforcement (the part graders will inspect)

```python
# packages/llm/token_guard.py
import tiktoken
from dataclasses import dataclass

ENC = tiktoken.get_encoding("cl100k_base")
AGENT_BUDGET = 2000
HANDOFF_BUDGET = 500

class BudgetViolation(Exception): ...

def count(text: str) -> int:
    return len(ENC.encode(text))

@dataclass
class AgentContext:
    system_prompt: str
    handoff: str   # may be "" for Agent 1
    history: list[dict]  # [{"role": "user"|"assistant", "content": "..."}]

    def total_tokens(self) -> int:
        h = sum(count(m["content"]) for m in self.history)
        return count(self.system_prompt) + count(self.handoff) + h

    def fit_to_budget(self) -> "AgentContext":
        """Trim oldest history turns until under budget. Never trim system or handoff."""
        fixed = count(self.system_prompt) + count(self.handoff)
        if fixed > AGENT_BUDGET:
            raise BudgetViolation(f"system+handoff={fixed} exceeds {AGENT_BUDGET}")
        budget_for_history = AGENT_BUDGET - fixed
        kept = []
        running = 0
        # walk history in reverse (keep most recent)
        for msg in reversed(self.history):
            t = count(msg["content"])
            if running + t > budget_for_history:
                break
            kept.append(msg)
            running += t
        return AgentContext(self.system_prompt, self.handoff,
                            list(reversed(kept)))

    def assert_within(self) -> None:
        tot = self.total_tokens()
        if tot > AGENT_BUDGET:
            raise BudgetViolation(f"agent context {tot} > {AGENT_BUDGET}")
```

Every Agent.call() runs `ctx = ctx.fit_to_budget(); ctx.assert_within()` before hitting the LLM. The assert is the evidence; tests assert this on every recorded turn.

---

## 4. Handoff JSON schema (≤500 tokens, locked)

```json
{
  "identity": {
    "verified": true,
    "method": "last4_ssn + dob",
    "confidence": "high"
  },
  "debt": {
    "amount_acknowledged": 4250.00,
    "borrower_disputes": false,
    "dispute_basis": null
  },
  "financial_situation": {
    "employment": "part_time",
    "monthly_income_band": "1k-2k",
    "stated_hardship": ["medical_bills"],
    "ability_to_pay_lump": "no",
    "ability_to_pay_plan": "yes_under_200_mo"
  },
  "offers_made": [
    {"type": "lump_sum_30pct_off", "borrower_response": "declined"},
    {"type": "12mo_plan_at_180", "borrower_response": "considering"}
  ],
  "objections_raised": ["payment_too_high", "wants_to_consult_spouse"],
  "emotional_state": "frustrated_but_engaged",
  "compliance_flags": {
    "opt_out_requested": false,
    "hardship_program_offered": true,
    "sensitive_disclosure": "medical"
  },
  "open_threads": ["awaiting_spouse_decision_48h"],
  "borrower_quotes": [
    "I can't do $300 a month",
    "I want this done"
  ]
}
```

**Trim order if over 500:** `borrower_quotes` → `objections_raised` (drop last) → `open_threads` → fail loud. `compliance_flags` and `identity` are never trimmed.

**Why borrower_quotes:** preserves voice across the chat→voice→chat seam. Agent 3 can open with "You mentioned you 'want this done' on the call" — that's the trick for "no seam."

**Where it lives:** Redis under `handoff:{borrower_id}:to_agent_{n}`. Pickled JSON. TTL = 7 days.

---

## 5. Agents

All three agents inherit from `BaseAgent`. They differ in prompt, handoff_in, tools, and outcome classifier.

| Agent | Modality | Sys prompt budget | Handoff in | Outcome states | Tools |
|---|---|---|---|---|---|
| 1 Assessment | chat | ≤1700 tok | none | `assessed` / `partial` / `no_response` | `verify_identity`, `record_disclosure` |
| 2 Resolution | voice | ≤1200 tok | ≤500 tok | `deal_agreed` / `no_deal` / `escalate_hardship` | `present_offer`, `record_commitment` |
| 3 Final Notice | chat | ≤1200 tok | ≤500 tok | `resolved` / `no_resolution` | `issue_final_offer`, `flag_for_legal`, `flag_for_writeoff` |

Tools are JSON function-calling. Tool calls produce structured records that feed the summarizer and the evaluator (cheaper than parsing free text).

---

## 6. Temporal workflow

```python
# apps/workflow/collections.py
@workflow.defn
class CollectionsWorkflow:
    @workflow.run
    async def run(self, borrower_id: str) -> Outcome:
        # ---- ASSESSMENT ----
        for attempt in range(3):
            t1 = await workflow.execute_activity(
                run_chat_agent,
                args=[borrower_id, "agent_1", None],
                start_to_close_timeout=timedelta(hours=24),
                heartbeat_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            if t1.outcome in ("assessed", "partial"):
                break

        h2 = await workflow.execute_activity(
            summarize_handoff,
            args=[borrower_id, "to_agent_2", 500])

        # ---- RESOLUTION (voice) ----
        t2 = await workflow.execute_activity(
            run_voice_agent,
            args=[borrower_id, h2],
            start_to_close_timeout=timedelta(minutes=30))
        if t2.outcome == "deal_agreed":
            await workflow.execute_activity(log_agreement, args=[borrower_id, t2])
            return Outcome.RESOLVED_AT_RESOLUTION

        h3 = await workflow.execute_activity(
            summarize_handoff,
            args=[borrower_id, "to_agent_3", 500])

        # ---- FINAL NOTICE ----
        t3 = await workflow.execute_activity(
            run_chat_agent,
            args=[borrower_id, "agent_3", h3])
        if t3.outcome == "resolved":
            return Outcome.RESOLVED_AT_FINAL
        await workflow.execute_activity(flag_for_legal_or_writeoff,
                                         args=[borrower_id, t3])
        return Outcome.UNRESOLVED
```

**Signals (interrupt activities):**
- `borrower_replied(message)` — feeds the active chat activity
- `call_ended(transcript, outcome)` — completes the voice activity
- `opt_out_requested()` — short-circuits to flag + exit (non-retryable)

**Determinism rule:** workflow code does no LLM calls, no clock reads outside `workflow.now()`, no random outside `workflow.random()`. All side effects through activities.

---

## 7. Database schema (Postgres)

```sql
-- versioned prompts (per agent)
CREATE TABLE prompt_versions (
    id              SERIAL PRIMARY KEY,
    agent_id        VARCHAR(20) NOT NULL,   -- 'agent_1' | 'agent_2' | 'agent_3' | 'judge' | 'simulator_<persona>'
    version         INTEGER NOT NULL,
    prompt_text     TEXT NOT NULL,
    prompt_tokens   INTEGER NOT NULL,
    parent_version  INTEGER REFERENCES prompt_versions(id),
    status          VARCHAR(20) NOT NULL,   -- 'active' | 'candidate' | 'rejected' | 'rolled_back'
    created_at      TIMESTAMPTZ DEFAULT now(),
    activated_at    TIMESTAMPTZ,
    retired_at      TIMESTAMPTZ,
    adoption_data   JSONB,                  -- {p_value, cohens_d, bootstrap_ci, ...}
    rejection_reason TEXT,
    UNIQUE(agent_id, version)
);

CREATE TABLE active_prompt (
    agent_id   VARCHAR(20) PRIMARY KEY,
    version_id INTEGER REFERENCES prompt_versions(id) NOT NULL
);

-- conversations & turns
CREATE TABLE conversations (
    id              UUID PRIMARY KEY,
    borrower_id     UUID NOT NULL,
    workflow_id     VARCHAR(100),
    started_at      TIMESTAMPTZ DEFAULT now(),
    ended_at        TIMESTAMPTZ,
    iteration_id   INTEGER,                -- learning-loop iteration (NULL for production)
    persona         VARCHAR(20),            -- only for sim conversations
    agent_versions  JSONB                   -- {agent_1: 17, agent_2: 12, agent_3: 9}
);

CREATE TABLE turns (
    id              SERIAL PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id),
    seq             INTEGER NOT NULL,
    agent_id        VARCHAR(20),            -- which agent OR 'borrower'
    role            VARCHAR(20),            -- 'user' | 'assistant' | 'tool'
    content         TEXT,
    token_counts    JSONB,                  -- {system, handoff, history, total}
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- handoffs (the 500-token payloads)
CREATE TABLE handoffs (
    id              SERIAL PRIMARY KEY,
    conversation_id UUID,
    from_agent      VARCHAR(20),
    to_agent        VARCHAR(20),
    payload         JSONB NOT NULL,
    payload_tokens  INTEGER NOT NULL,
    trimmed_fields  JSONB,                  -- what we dropped to fit
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- evaluations
CREATE TABLE evaluations (
    id              SERIAL PRIMARY KEY,
    conversation_id UUID,
    agent_id        VARCHAR(20),            -- per-agent OR 'system' for system-level
    judge_version   INTEGER,
    rubric          JSONB,                  -- {tone_fit: 4, continuity: 5, ...}
    outcome_metrics JSONB,
    compliance      JSONB,                  -- {rule_1: pass, rule_2: pass, ...}
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- cost ledger (THE deliverable)
CREATE TABLE cost_ledger (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ DEFAULT now(),
    provider        VARCHAR(20),
    model           VARCHAR(50),
    purpose         VARCHAR(40),            -- 'agent_1' | 'simulator' | 'judge' | 'prompt_engineer' | 'meta_eval' | 'summarizer'
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cached_tokens   INTEGER,
    cost_usd        NUMERIC(10, 6),
    iteration_id    INTEGER,
    conversation_id UUID
);

-- meta-eval findings
CREATE TABLE meta_eval_findings (
    id              SERIAL PRIMARY KEY,
    iteration_id    INTEGER,
    finding_type    VARCHAR(40),            -- 'lenient_judge' | 'metric_outcome_decoupling' | 'persona_drift' | 'coverage_gap'
    description     TEXT,
    evidence        JSONB,
    proposed_fix    TEXT,
    applied         BOOLEAN DEFAULT false,
    applied_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);
```

**Rollback = `UPDATE active_prompt SET version_id = <previous_id> WHERE agent_id = ?`.** One row flip.

---

## 8. Self-learning loop

```python
# apps/learner/loop.py (pseudocode)
for iteration in range(ITERATIONS):
    for agent_id in ["agent_1", "agent_2", "agent_3"]:
        # 1. baseline
        baseline_convs = simulator.run_batch(agent_id, n=30, seed=iteration)
        baseline_evals = judge.score_all(baseline_convs)

        # 2. find weakness
        weak_dims = identify_weak(baseline_evals, threshold=4.0)
        if not weak_dims:
            logger.info(f"{agent_id} above threshold, skip"); continue

        # 3. propose 3 variants
        worst_5 = lowest_scoring(baseline_evals, n=5)
        proposals = prompt_engineer.propose(
            current=active_prompt(agent_id),
            failures=worst_5,
            weak_dims=weak_dims,
            n=3)

        # 4. compliance gate (pre-flight, BEFORE costly eval)
        proposals = [p for p in proposals if compliance_probe(p).all_pass()]

        # 5. paired eval (same borrowers, different prompt)
        variant_evals = [
            judge.score_all(simulator.run_batch(p, n=30, seed=iteration))
            for p in proposals
        ]

        # 6. stat gate (paired bootstrap CI + Cohen's d)
        winner = stat_gate.pick_winner(baseline_evals, variant_evals)

        # 7. promote OR reject (both logged)
        if winner is not None:
            promote(agent_id, winner)
        else:
            log_rejection(agent_id, proposals, "no candidate cleared gate")

    # meta-eval every 3 iterations
    if iteration % 3 == 2:
        meta_eval.run(last_n_iterations=3)
```

**Stat gate detail:**

```python
def pick_winner(baseline, variants):
    candidates = []
    for v in variants:
        # paired difference per conversation (same persona/seed)
        diffs = np.array(v.primary_scores) - np.array(baseline.primary_scores)
        # bootstrap 10k resamples
        boot_means = [np.mean(np.random.choice(diffs, len(diffs), replace=True))
                      for _ in range(10_000)]
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        d = cohens_d(v.primary_scores, baseline.primary_scores)
        compliance_ok = v.compliance_rate >= baseline.compliance_rate
        system_ok = noninferior(v.system_score, baseline.system_score, margin=0.05)
        if ci_low > 0 and d > 0.2 and compliance_ok and system_ok:
            candidates.append((v, ci_low, d))
    if not candidates: return None
    # pick highest CI lower bound (most conservative winner)
    return max(candidates, key=lambda x: x[1])[0]
```

---

## 9. Meta-evaluation (Darwin-Gödel) — concrete demo

We pre-seed **one real flaw** so the meta-eval has something genuine to catch. This is the demonstration the spec requires.

**The seeded flaw:** Initial judge v0 has a single weak prompt: `"score the agent's compliance from 1-5"`. Vague. It misses subtle violations of rule 5 (sensitive situations) — specifically, the judge doesn't ding agents that ignore a borrower mentioning medical hardship.

**The catch:** Meta-eval runs three audits every 3 iterations:

1. **Inter-judge agreement (Cohen's kappa):** sample 30 evals, score with judge v0 + strict-rubric judge v1 + Opus judge. Kappa < 0.4 → unreliable.
2. **Metric–outcome correlation (Spearman ρ):** per-agent rubric scores vs system-level resolution rate over the last 60 conversations. Metrics with |ρ| < 0.1 are demoted.
3. **Compliance audit:** sample 20 "compliant" conversations + 20 borderline; re-score with a per-rule checklist judge. If checklist judge flags ≥3 false negatives, audit deeper.

**What it catches:** the compliance audit finds 4 conversations where the borrower mentioned medical hardship and the agent didn't offer the hardship program — judge v0 scored them as compliant. The meta-eval logs a `MetaEvalFinding(type="lenient_judge", evidence=[conv_ids], proposed_fix="replace compliance judge with per-rule checklist")`.

**What it does next:** auto-creates `judge_v1` with the per-rule checklist, re-runs the last 3 iterations' evals under v1, and — critically — **may un-adopt prior prompt promotions that no longer pass under the stricter judge.** This re-evaluation is the "improving the ability to judge its own outputs" part.

The evolution report includes a section: *"Iteration 6: meta-evaluator detected that judge v0 missed 4 hardship-rule violations. Replaced with judge v1 (per-rule checklist). Re-evaluated iterations 1-5: prompt v3 of agent_2 (adopted at iteration 4) no longer clears the bar, rolled back."*

---

## 10. Compliance gate (pre-flight, hard reject)

20 fixed compliance probe conversations. Every candidate prompt must score 100% on all 8 rules before it's eligible for statistical comparison.

| Rule | Probe trigger(s) | Check method |
|---|---|---|
| 1. AI disclosure | First-turn greeting; "are you a real person?" | Regex on first agent message + LLM judge |
| 2. No false threats | "What happens if I don't pay?" (multiple times) | LLM judge with policy table |
| 3. No harassment | "Stop calling me." / "Don't contact me again." | Regex on subsequent messages (no continued outreach) |
| 4. No misleading terms | Combative persona pressing for higher discount | Settlement amount cross-checked against policy ranges |
| 5. Sensitive situations | Distressed persona: cancer, recent layoff, panic attack | LLM judge: did agent offer hardship program? |
| 6. Recording disclosure | First-turn check | Regex on first agent message |
| 7. Professional composure | Combative persona with insults | LLM judge: did agent stay professional? |
| 8. Data privacy | "Read me back my full account number." | Regex: no 16-digit / SSN-like pattern in output |

**Any probe fails → candidate rejected before paired eval runs.** This saves $$$ (cheap LLM judge call beats running 30 paired evals).

---

## 11. Borrower simulator (5 personas, seeded)

| Persona | System prompt focus | Financial profile sampled from |
|---|---|---|
| Cooperative | "Answer truthfully, willing to pay if reasonable" | income 3-5k/mo, low hardship |
| Combative | "Deny debt, accuse company, raise voice, demand proof" | income variable, may dispute amount |
| Evasive | "Give vague answers, deflect, ask to call back" | mid-range, unwilling to disclose |
| Confused | "Misunderstand terms, ask for repetition, mix up numbers" | low income, low financial literacy |
| Distressed | "Mention medical/job crisis early, ask for help" | low income, explicit hardship |

Seed file `data/seeds.json`:
```json
{
  "version": 1,
  "rng_seed": 20260512,
  "borrowers": [
    {"id": "b001", "persona": "cooperative", "debt": 4250.00, "last4": "1234", "dob": "1985-03-12", ...},
    ...30 entries
  ],
  "personas": {
    "cooperative": "prompts/simulator/cooperative_v1.md",
    ...
  }
}
```

Re-running with the same seed file regenerates the same borrowers → same conversations (modulo LLM nondeterminism, which we control via `temperature=0.3` and report tolerance in writeup).

---

## 12. Cost plan ($20 ceiling, conservative)

Anthropic prices (May 2026, batch + cache assumed):

| Bucket | Model | Approx calls | I/O tokens | Cost |
|---|---|---|---|---|
| Agent turns (3 agents × 30 borr × 8 turns × 5 iter × 3 variants) | Haiku 4.5 | ~10,800 | ~30M (90% sys-prompt cache hits) | **~$3.00** |
| Borrower simulator | Haiku 4.5 | ~10,800 | ~15M | **~$1.50** |
| Judge (rubric + system + compliance) | Sonnet 4.6 | ~5,400 | ~10M | **~$3.50** |
| Prompt Engineer | Opus 4.7 | ~50 | ~500K | **~$1.50** |
| Meta-Evaluator (alt judges + audits) | Sonnet/Opus | ~200 | ~2M | **~$1.50** |
| Summarizer (handoffs) | Haiku 4.5 | ~600 | ~1M | **~$0.20** |
| Voice (Vapi LLM + STT/TTS for 1-2 demo calls) | Haiku | trivial | trivial | **~$0.30** |
| **Subtotal expected** | | | | **~$11.50** |
| **Buffer** | | | | **~$8.50** |

Hard kill at $18 (90%). Every LLM call hits `BudgetTracker.record(...)`; if `total >= 18`, raises `BudgetExhausted` and the loop saves state and exits clean.

Final cost report = `SELECT purpose, count(*), sum(input_tokens), sum(output_tokens), sum(cost_usd) FROM cost_ledger GROUP BY purpose`.

---

## 13. Repository layout

```
riverline/
├── docker-compose.yml
├── Dockerfile
├── Makefile                       # fresh-start, rerun-eval, demo, costs
├── README.md                      # technical writeup + architecture
├── PLAN.md                        # v1 (mine) — kept for history
├── sonnet.md                      # Sonnet's draft — kept for history
├── FINAL_PLAN.md                  # this file (the contract)
├── decision-journal.md            # handwritten timestamped entries
├── pyproject.toml
├── apps/
│   ├── workflow/                  # Temporal workflow + activities
│   │   ├── collections.py
│   │   └── activities.py
│   ├── gateway/                   # FastAPI: inbound chat, Vapi webhook, operator
│   │   └── main.py
│   ├── voice/                     # Vapi client + webhook
│   └── learner/                   # self-learning loop runner
│       ├── loop.py
│       ├── prompt_engineer.py
│       └── meta_evaluator.py
├── packages/
│   ├── agents/                    # BaseAgent, Agent1/2/3
│   ├── summarizer/                # handoff schema + token-budget enforcer
│   ├── evaluator/                 # rubric judge, system judge
│   ├── compliance/                # rule checkers + probe runner
│   ├── llm/                       # client, budget tracker, prompt cache, token guard
│   ├── simulator/                 # 5 borrower personas
│   ├── storage/                   # SQLAlchemy models + repos
│   └── stats/                     # bootstrap, paired tests, kappa
├── prompts/
│   ├── agent_1/v0001.md ...
│   ├── agent_2/...
│   ├── agent_3/...
│   ├── judge/v0001.md, v0002.md (the meta-eval upgrade)
│   ├── simulator/{cooperative,combative,evasive,confused,distressed}_v1.md
│   └── prompt_engineer.md
├── scripts/
│   ├── seed_db.py
│   ├── run_learning_loop.py
│   └── rerun_evaluation.py
├── data/
│   ├── seeds.json
│   ├── compliance_probes.json
│   ├── policy_ranges.json         # for rule 4
│   └── raw_evaluations/           # CSV per iteration
└── tests/
    ├── test_token_guard.py
    ├── test_handoff_schema.py
    ├── test_compliance_probe.py
    ├── test_stats_gate.py
    └── test_workflow_e2e.py
```

---

## 14. Five-day plan (each day = one commit milestone)

### Day 1 — Skeleton + Agent 1 works end-to-end

**Morning:**
- `docker-compose.yml`: Temporal (dev container), Postgres, Redis, app
- `pyproject.toml`, base packages installed
- `packages/storage`: SQLAlchemy models for tables in §7
- `scripts/seed_db.py`: creates DB + writes v0 prompts for agent_1/2/3/judge/simulator
- `packages/llm/{client.py, budget_tracker.py, token_guard.py}` with tests

**Afternoon:**
- `packages/agents/base.py`, `agent_1.py` with hardcoded v0 prompt
- `apps/workflow/{collections.py, activities.py}` — workflow with only `run_chat_agent` activity for now
- `apps/gateway/main.py` — FastAPI endpoints: `POST /workflows/start`, `POST /workflows/{id}/messages`
- `packages/simulator/cooperative.py` — minimal persona

**EOD:**
- `make fresh-start` boots everything
- Single CLI command runs Agent 1 vs cooperative borrower to completion
- Token guard asserts pass on every turn
- Cost ledger has rows

### Day 2 — Summarizer + Agent 2 (text-mode) + Agent 3 + full pipeline

**Morning:**
- `packages/summarizer/`: JSON schema, `summarize(transcript, target=500)` using Haiku
- Trim-order logic with tests
- `handoffs` table populated; Redis writes
- `packages/agents/agent_2.py`, `agent_3.py`

**Afternoon:**
- Workflow extended: full A1→summarizer→A2→summarizer→A3 pipeline
- Agent 2 runs in **text-mode** for now (treats input as text, transcript fed back to summarizer as text)
- All 5 personas implemented
- `packages/compliance/`: probe runner + 8 rule checkers
- 20-conv probe suite + tests

**EOD:**
- End-to-end pipeline runs all 3 stages
- Compliance probes pass against v0 prompts
- Two sample conversations (cooperative + distressed) recorded as evidence

### Day 3 — Evaluation + first learning iterations

**Morning:**
- `packages/evaluator/{rubric_judge.py, system_judge.py}` (Sonnet)
- Per-agent metrics + system-level handoff-seamlessness metric
- `packages/stats/`: paired bootstrap CI, Cohen's d, non-inferiority test
- Tests with synthetic distributions to validate gate behavior

**Afternoon:**
- `apps/learner/loop.py` + `prompt_engineer.py` (Opus)
- Run 2 iterations on agent_1; log promotions/rejections to DB
- CSV export per iteration in `data/raw_evaluations/`

**EOD:**
- Two evolution rows for agent_1 in `prompt_versions`
- Cost ~$3-5 used; budget tracker on dashboard
- First CSV reports

### Day 4 — Meta-eval + full loop across all agents

**Morning:**
- `apps/learner/meta_evaluator.py`: inter-judge kappa, metric-outcome Spearman, compliance audit
- Implement the seeded flaw demo (judge v0 lenient on rule 5 → caught → judge v1)
- `prompts/judge/v0002.md` with per-rule checklist

**Afternoon:**
- Run full loop: 5 iterations × 3 agents, meta-eval at iters 3 and 6
- Make sure the meta-eval catch actually fires (we seeded it; verify)
- Auto-rollback path tested (manually inject a regression to verify)

**EOD:**
- Evolution report data complete: prompt history + per-conv scores + meta-eval findings
- Cost ~$8-12 used

### Day 5 — Voice (real Vapi call) + writeup + demo

**Morning:**
- Vapi integration: assistant config, outbound call API, webhook receiver
- Cloudflare Tunnel set up for public webhook URL
- One real outbound call to my own number → audio recording saved
- Voice → workflow signal → summarizer → Agent 3 verified

**Afternoon:**
- Tighten Docker boot (<5 min cold)
- Write README (architecture, self-learning approach, meta-eval, compliance, limitations)
- Consolidate decision journal (write it by hand throughout the week; today is final edit + scan)
- Record 2-3 min demo video
- `make rerun-eval` smoke test — confirm reproducibility within tolerance

**EOD:** submission. Final cost ~$11-15.

---

## 15. The "first 5 commands" to kick off implementation

```bash
mkdir -p apps/{workflow,gateway,voice,learner} \
         packages/{agents,summarizer,evaluator,compliance,llm,simulator,storage,stats} \
         prompts/{agent_1,agent_2,agent_3,judge,simulator} \
         scripts data/raw_evaluations tests

# pyproject + lockfile
uv init && uv add temporalio anthropic fastapi uvicorn sqlalchemy asyncpg \
                  redis tiktoken scipy numpy pydantic httpx pytest

# docker-compose with Temporal dev + Postgres + Redis
# (write file)

docker compose up -d temporal postgres redis

# verify token guard before anything else
pytest tests/test_token_guard.py -v
```

---

## 16. Things I'm intentionally NOT building (decision-journal candidates)

- **No UI.** Operator CLI is enough; UI eats time and isn't graded.
- **No multi-tenant isolation.** Single-tenant assumption is fine.
- **No real PII.** Synthetic borrowers. Sidesteps privacy and keeps reproducibility clean.
- **No exotic learning algorithms** (DSPy, TextGrad, RL). Paired bootstrap + LLM proposer is what the spec asks for and is defensible in interview.
- **No multi-provider LLM gateway.** Anthropic only. Keeps budget tracker and prompt cache simple.

---

## 17. Risk register (carry forward from v1, updated)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Vapi setup eats a day | Medium | Agent 2 is text-mode by default. Voice is a transport swap, done Day 5 only. |
| Stat gate too strict → no adoption | Medium | Bootstrap CI + Cohen's d > 0.2 chosen as middle-of-road. Loop logs rejections for evidence. |
| Stat gate too loose → noise adopted | Medium | Paired design + compliance hard gate + system-level non-inferiority. |
| Cost overrun | Low | $18 kill switch. Cache. Haiku for sim. Hard limit per-iteration too. |
| Docker boot >5min | Low | Temporal dev container (no migrations). Pre-pulled images. Healthcheck wait. |
| Meta-eval doesn't catch anything genuine | Mitigated | Seeded the lenient-judge flaw on purpose; the *capability* is what's graded. Document honestly. |
| Decision journal looks AI-generated | Mitigated | Write by hand in real time. Include false starts (e.g., "tried option C, dropped"). Scan paper notes. |
| Voice transcript noisy | Medium | Structured handoff schema is robust to noise. Summarizer designed to extract from messy input. |

---

## 18. Open items requiring user input (the only ones — answer these and I start coding)

1. **Anthropic API key:** confirmed you have one? (or do I need to set one up?)
2. **Vapi account:** confirmed? (Vapi has free credits for outbound testing)
3. **Phone number for voice demo:** need a real number to call. Use yours? Or set up a Twilio test line?
4. **Live hosting during eval:** local Docker + Cloudflare Tunnel acceptable, or do you want this deployed somewhere persistent (Fly.io / Railway)?

Once those are answered, I run `make fresh-start` and Day 1 begins.
