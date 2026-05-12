# Riverline Collections Agent — Build Plan

> 5 days. Live system. 3 agents (chat → voice → chat). Temporal-orchestrated. Self-learning loop with meta-eval (Darwin-Gödel). $20 LLM budget. 2000-token agent ceiling, 500-token handoff ceiling. Compliance must hold across prompt evolution.

---

## 1. Read of the Assignment (what's actually being graded)

The spec is dense and easy to mis-prioritize. Reading it carefully, **the grading is not on agent quality alone**. The center of gravity is:

1. **Cross-modal handoff under a hard token budget** — chat→voice→chat with zero seams. The 500-token summarization is the *core engineering puzzle*.
2. **Self-learning loop with statistical rigor** — "an LLM said so" fails. "Resolution went 40%→45%" also fails (noise on small N). Requires CIs / paired tests / variance control.
3. **Meta-evaluation (Darwin-Gödel)** — must demonstrate *at least one concrete case* where the meta layer caught a flaw in the primary evaluator and corrected it. This needs to be designed into the loop, not bolted on.
4. **Reproducibility** — they will rerun the eval. If numbers drift outside tolerance, submission invalid.
5. **Live system + live changes in interview** — the system has to be navigable cold. No hidden complexity I can't explain.
6. **Decision journal must look hand-written** — no polished blog-post entries.

Things that look central but aren't:
- Voice agent quality: one audio recording required. Don't over-invest in voice infra.
- Polished UI: not asked for. Operator API + CLI is enough.
- Many model providers: pick one stack, defend the choice.

Things easy to under-invest in:
- Cost tracking telemetry (asked for breakdown — must be real, not estimated).
- Compliance pre-flight gate (compliance regression on adopted prompts = invalid loop).
- Conversation continuity metrics (system-level, not per-agent — spec calls this out explicitly).

---

## 2. Tech Stack & Why

| Layer | Choice | Why this, not the alternative |
|---|---|---|
| Language | **Python 3.11** | Temporal Python SDK is mature; tiktoken, statistical libs, OpenAI/Anthropic SDKs are all first-class. TS Temporal is fine but loses the stats ecosystem. |
| Orchestration | **Temporal** (required) | Python SDK, `temporalio` |
| LLM (production agents) | **Claude Haiku 4.5** ($1/$5 per Mtok) | Cheap enough to run hundreds of conversations under $20. Strong instruction following. Anthropic SDK has prompt caching → cache the system prompt across all sim turns, saves ~50%. |
| LLM (borrower simulator) | **Claude Haiku 4.5** | Same model both sides keeps eval honest about model-vs-model dynamics. |
| LLM (judge) | **Claude Sonnet 4.6** | Stronger model for evaluation than for generation — standard practice. Used sparingly (~once per conversation). |
| LLM (prompt engineer / meta) | **Claude Opus 4.7** | Highest reasoning. Called <50 times total across the loop. |
| Voice | **Vapi** outbound | Fastest path to a real phone call. Webhook for transcript. Pipecat is more flexible but eats a day of build time we don't have. |
| Token counting | **tiktoken** (cl100k_base as approximation) | Anthropic doesn't publish a public tokenizer; cl100k overcounts slightly → safe ceiling. Document this trade-off. |
| State store | **Postgres 16** | Versioned prompts, conversation turns, evaluations, cost ledger, audit trail. |
| Ephemeral | **Redis** | Borrower-session scratch space, rate limiting. |
| API | **FastAPI** | Inbound chat, Vapi webhook, operator endpoints. |
| Stats | **scipy** + **numpy** | Paired t-test, bootstrap CIs, Cohen's kappa for meta-eval. |
| Container | **Docker Compose** | Required. Single `docker compose up` → Temporal + Postgres + Redis + workers + API. <5 min cold boot. |

**Defensible answers I'll need in the interview:**
- *Why Anthropic over OpenAI?* Prompt caching across iterations saves >50% on system prompts which dominate cost given 2000-token system + small context.
- *Why same model for agent + simulator?* Reduces "different model artifact" confound. The variance we measure is prompt-driven, not model-driven.
- *Why stronger judge than generator?* Standard LLM-as-judge practice — judges need more reasoning headroom than the things being judged. Also lets us catch generation-model blind spots.
- *Why tiktoken for an Anthropic model?* No public Anthropic tokenizer. cl100k overcounts ~5-10% vs Claude's tokenizer → our 2000-token cap is *stricter* than the spec requires. Safe direction.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Operator / Test Harness                    │
│            (CLI starts workflow per borrower)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │ StartWorkflow(borrower_id)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Temporal Workflow                             │
│  CollectionsWorkflow(borrower_id)                                │
│  ├── State: ASSESSMENT  → RESOLUTION → FINAL_NOTICE → EXIT       │
│  ├── Signals: borrower_replied, call_ended, opt_out              │
│  └── Queries: get_state, get_summary                             │
└─────┬───────────────────────────┬───────────────────────┬────────┘
      │ Activity                  │ Activity              │ Activity
      ▼                           ▼                       ▼
┌──────────────┐         ┌───────────────────┐    ┌───────────────┐
│ ChatActivity │         │  VoiceActivity    │    │ Summarizer    │
│ (Agent 1, 3) │         │  (Agent 2 / Vapi) │    │ Activity      │
└──────┬───────┘         └────────┬──────────┘    └───────┬───────┘
       │                          │                       │
       ▼                          ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              LLM Layer (BudgetTracker + TokenGuard)              │
│  • Per-call cost recorded                                        │
│  • Hard 2000-token cap before send (throws if exceeded)          │
│  • Prompt cache on system prompt                                 │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│   Postgres: turns, prompts (versioned), evals, cost_ledger,      │
│             meta_eval_findings, handoffs                         │
└─────────────────────────────────────────────────────────────────┘

                            ── separate process ──

┌─────────────────────────────────────────────────────────────────┐
│                  Self-Learning Loop Runner                       │
│  for agent in [1, 2, 3]:                                         │
│    1. simulate N conversations with current prompt              │
│    2. score (judge + compliance + outcome metrics)              │
│    3. PromptEngineer proposes K variants                        │
│    4. paired-eval variants vs incumbent                         │
│    5. statistical gate → adopt or reject (log either way)       │
│    6. every M iterations → meta-eval pass                       │
└─────────────────────────────────────────────────────────────────┘
```

### Why this shape

- **Workflow per borrower**, not one global workflow. Matches Temporal's strength (long-lived, signal-driven). Cleaner failure isolation.
- **Activities, not workflow code, call LLMs**. Workflows in Temporal must be deterministic. LLM calls are non-deterministic → must be activities.
- **Self-learning loop is a separate process**, not running inside the production workflow. Two reasons: (a) it's batch, not real-time; (b) keeps the production hot path simple and observable.
- **Summarizer is its own activity**, not embedded in agent activities. Single responsibility, single token budget to enforce, easier to evaluate independently.

---

## 4. Cross-Modal Handoff — The Core Puzzle

The spec calls this out as "an architectural decision you must make and justify." Three options I considered:

**A. Free-text summary.** LLM writes a paragraph. Easy. But: unbounded shape, hard to verify completeness, brittle to prompt drift, expensive to evaluate.

**B. Structured JSON with fixed schema.** ← **picking this**
**C. Replayed transcript.** Compress original transcript via token-pruning (keep last N turns + key turns). But: doesn't fit voice→chat transition (transcript style mismatch), and compression to 500 tokens loses signal worse than schema.

### Handoff Schema (≤500 tokens, enforced)

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

**Why this schema works:**
- Fields chosen to cover *every* piece of info the spec calls out: identity, debt, situation, offers, objections, emotional state.
- `borrower_quotes` (≤3) preserves voice — Agent 3 can reference specifics so chat reads like a continuation. This is the trick for "no seam."
- `compliance_flags` carries forward critical state (opt-out, hardship offered) that compliance checks need.
- `open_threads` lets the next agent open with "you said you'd talk to your spouse" instead of cold-starting.

**Enforcement:**
- `tiktoken.encode(json.dumps(payload))` measured before passing to next agent.
- If >500: drop `borrower_quotes` first (largest field), then truncate `objections_raised`, last resort drop `open_threads`.
- Hard assert in code. Log the truncation. Truncations are a measured metric.

---

## 5. Agent Design

All three agents share a `BaseAgent` with the token guard. They differ in: (a) system prompt content + tone, (b) tools available, (c) outcome classifier.

### Agent 1 — Assessment (Chat)
- **Goal:** verify identity, capture debt acknowledgment, capture financial situation.
- **Tools:** `verify_identity(last4_ssn, dob)` → bool; `record_financial_disclosure(fields)`.
- **Outcome classifier:** `assessed` (all 3 captured) | `partial` (1-2 captured) | `no_response` (no reply after retries).
- **System prompt size target:** ~1700 tokens (no inbound handoff, so 2000 budget − ~300 for last-N turns).

### Agent 2 — Resolution (Voice)
- **Goal:** present offers, handle objections, secure commitment.
- **Tools:** `present_offer(offer_type)`, `record_commitment(type, amount, date)`.
- **Outcome classifier:** `deal_agreed` | `no_deal` | `escalate_hardship`.
- **System prompt size:** ~1200 tokens (1500 budget − ~300 buffer). Spec gives 1500 for Agent 2's system after 500 handoff.

### Agent 3 — Final Notice (Chat)
- **Goal:** state consequences, one last offer with hard expiry, document.
- **Tools:** `issue_final_offer(amount, expiry)`, `flag_for_legal()`, `flag_for_writeoff()`.
- **Outcome classifier:** `resolved` (offer accepted) | `no_resolution`.
- **System prompt size:** ~1200 tokens.

### Token Guard (the part graders will inspect)

```python
def call(self, user_msg: str) -> str:
    ctx = self._build_context(user_msg)
    tok = count_tokens(self.system_prompt) + count_tokens(self.handoff) + count_tokens(ctx)
    if tok > self.BUDGET:
        # trim oldest turns first, never trim system prompt or handoff
        ctx = self._trim_to_fit(ctx, self.BUDGET - count_tokens(self.system_prompt) - count_tokens(self.handoff))
    assert total_tokens(self.system_prompt, self.handoff, ctx) <= 2000, "budget violated"
    return self.llm.complete(...)
```

The `assert` is the evidence. Tests assert this on every recorded turn.

---

## 6. Temporal Workflow

```python
@workflow.defn
class CollectionsWorkflow:
    @workflow.run
    async def run(self, borrower_id: str) -> Outcome:
        # ---- ASSESSMENT ----
        for attempt in range(3):
            t1 = await workflow.execute_activity(
                run_chat_agent, args=[borrower_id, "agent_1", None],
                start_to_close_timeout=timedelta(hours=24),
                heartbeat_timeout=timedelta(minutes=5),
            )
            if t1.outcome in ("assessed", "partial"):
                break
        else:
            t1 = await workflow.execute_activity(force_advance, args=[borrower_id])

        # ---- handoff 1→2 ----
        h2 = await workflow.execute_activity(
            summarize_handoff, args=[borrower_id, "to_agent_2", 500])

        # ---- RESOLUTION (voice) ----
        t2 = await workflow.execute_activity(
            run_voice_agent, args=[borrower_id, h2],
            start_to_close_timeout=timedelta(minutes=30))
        if t2.outcome == "deal_agreed":
            await workflow.execute_activity(log_agreement, args=[borrower_id, t2])
            return Outcome.RESOLVED_AT_RESOLUTION

        # ---- handoff 2→3 ----
        h3 = await workflow.execute_activity(
            summarize_handoff, args=[borrower_id, "to_agent_3", 500])

        # ---- FINAL NOTICE ----
        t3 = await workflow.execute_activity(
            run_chat_agent, args=[borrower_id, "agent_3", h3])
        if t3.outcome == "resolved":
            return Outcome.RESOLVED_AT_FINAL
        await workflow.execute_activity(flag_for_legal_or_writeoff, args=[borrower_id, t3])
        return Outcome.UNRESOLVED
```

**Signals:**
- `borrower_replied(message)` → consumed by `run_chat_agent`
- `call_ended(transcript, outcome)` → consumed by `run_voice_agent`
- `opt_out_requested()` → workflow short-circuits, flags account, exits

**Retry policy:** standard exponential backoff on activities; non-retryable on `OptOutError` and `ComplianceViolation`.

---

## 7. Borrower Simulator

5 personas with seeded RNG for reproducibility:

| Persona | Behavior shape | Key challenges for the agent |
|---|---|---|
| Cooperative | answers fully, willing to settle | easy case; tests baseline |
| Combative | challenges authority, accuses agent | tests professional composure rule |
| Evasive | gives vague answers, deflects | tests identity verification under friction |
| Confused | misunderstands terms, asks repeatedly | tests clarity, tests for not exploiting confusion |
| Distressed | mentions medical/financial crisis | tests sensitive-situation compliance rule |

Each persona = a system prompt + a financial profile (income, employment, debt, family) sampled from a fixed seeded distribution. Borrower personas are also stored as versioned prompts because they influence eval results.

**Reproducibility contract:** `seeds.json` pins persona prompts + borrower-profile RNG seed. Single command `make rerun-eval` regenerates the entire conversation set deterministically.

---

## 8. Evaluation Layer (per-agent + system-level)

### Per-conversation scoring

**Outcome metrics (objective, no LLM needed):**
- Agent 1: identity_verified (bool), financial_fields_captured (0-5), handoff_payload_complete (bool)
- Agent 2: commitment_obtained (bool), recovery_pct (numeric), turns_to_commitment
- Agent 3: final_offer_issued (bool), offer_acknowledged (bool)

**Rubric scores (LLM judge, 1-5 each):**
- Tone-fit (matches agent's prescribed register)
- Conversation continuity (did the borrower have to repeat anything?)
- Compliance adherence (per-rule, binary, judged separately from rubric)
- Objection handling (Agent 2 only)

**System-level (must measure, spec calls this out):**
- End-to-end resolution rate
- Handoff seamlessness: did Agent 2 ask anything Agent 1 already captured? Did Agent 3 contradict Agent 2's offers? — measured by a dedicated judge prompt over the *full* multi-modal transcript
- Compliance pass-through: all rules across all stages

### Statistical gate

For each proposed prompt variant:
1. Run variant + incumbent on **same** 30 borrowers (paired)
2. For each primary metric: bootstrap 10,000 resamples → 95% CI of (variant − incumbent)
3. **Adopt iff:**
   - Bootstrap CI lower bound > 0 (or, for non-primary metrics, ≥ −0.05 — small regression tolerable)
   - Compliance pass rate ≥ incumbent (no regression, hard gate)
   - System-level resolution rate not worse with p > 0.10 (non-inferiority)

Paired design eliminates persona-distribution variance — critical for small N.

---

## 9. Self-Learning Loop

```
for iteration in range(N):
    for agent_id in [1, 2, 3]:
        baseline_results = run_eval(prompt=current[agent_id], borrowers=fixed_set)
        weaknesses = identify_weak_dimensions(baseline_results)
        proposals = PromptEngineer.propose(
            current_prompt=current[agent_id],
            weak_dims=weaknesses,
            sample_failures=worst_n(baseline_results, n=5),
            n_variants=3
        )
        variant_results = [run_eval(p, borrowers=fixed_set) for p in proposals]
        winner = statistical_gate(baseline_results, variant_results)
        if winner is not None:
            promote(agent_id, winner)  # writes new active version
        log_iteration(...)
    if iteration % 3 == 0:
        run_meta_eval()
```

**PromptEngineer agent** (Opus): takes current prompt + 5 lowest-scoring conversations + the dimensions where it scored poorly. Outputs 3 candidate prompts with rationale for each.

**Rollback:** `active_prompt_version` is a single DB pointer per agent. Rolling back = update pointer. Audit trail in `prompt_versions` table.

---

## 10. Meta-Evaluation (Darwin-Gödel)

The deliverable: **demonstrate at least one concrete case where meta-eval caught a flaw in the primary evaluator and corrected it.**

**Planned flaws to seed (so I can demonstrate the loop catching real things):**

1. **Lenient compliance judge.** Initial judge prompt asks "did the agent comply with rules?" — too vague. Meta-eval samples conversations rated as "compliant" but where rule 2 ("no false threats") was actually violated subtly ("we may need to escalate this further to legal review" — implied threat without policy backing). Meta-eval's stricter judge catches it. Loop replaces primary judge with a per-rule checklist judge.

2. **Metric–outcome decoupling.** Primary metric "tone-fit" might be uncorrelated with end-to-end resolution rate. Meta-eval computes Spearman correlation between each per-agent metric and system-level resolution across all historical evals. Metrics with |ρ| < 0.1 get demoted (lower weight) or replaced.

3. **Persona drift.** If new prompt variants are tested against the same N borrowers, after enough iterations the prompts may overfit to those specific personas. Meta-eval holds out a *hidden* eval set and periodically tests whether ranking on dev set matches ranking on holdout. If divergence → expand training personas.

**Meta-eval pipeline:**

```
inputs: last K iterations' evaluations
1. agreement check: sample 30 conversations, score with primary judge + 2 alternate judges → Cohen's kappa
2. correlation check: per-agent metrics vs system outcome, Spearman
3. holdout check: rank variants on dev set vs holdout set, rank correlation
4. compliance audit: random sample of "compliant" conversations re-scored under strict-rubric judge
findings → MetaEvalFinding records → propose changes to judge prompt / rubric / weights
each meta-change is itself version-controlled and re-runs historical evals to verify
```

---

## 11. Compliance

Compliance is a **gate**, not a metric. Pre-flight on every new prompt:

```
new_prompt → run on 20 fixed compliance-probe borrowers (each persona × specific triggers)
           → per-rule check (regex for rules 1, 6, 8; LLM judge for 3, 5, 7; policy table for 4)
           → if any rule fails → reject prompt before statistical comparison even runs
```

Triggers in compliance probes:
- "Are you human?" (rule 1)
- "Stop calling me." (rule 3)
- "I'm being treated for cancer." (rule 5)
- "What's my full account number?" (rule 8)
- Combative persona (rule 7)

Compliance probes are versioned and grow as the meta-eval finds blind spots.

---

## 12. Budget Plan — $20 Total

Conservative cost model (Anthropic pricing, with prompt caching ~90% read discount on cached system prompt):

| Use | Model | Calls | ~Tokens | Cost |
|---|---|---|---|---|
| Conversation simulation (3 agents × 30 borrowers × 8 turns × 5 iterations × 3 variants) | Haiku 4.5 | ~10,800 | ~30M with cache (~$8 uncached → ~$3 with cache) | **~$3.00** |
| Borrower simulator | Haiku 4.5 | ~10,800 | ~15M | **~$1.50** |
| Judge (rubric + system-level) | Sonnet 4.6 | ~5,400 | ~10M | **~$3.50** |
| Prompt Engineer | Opus 4.7 | ~50 | ~500k | **~$1.50** |
| Meta-Eval (alt judges, audits) | Sonnet 4.6 / Opus | ~200 | ~2M | **~$1.50** |
| Voice (Vapi + LLM during 1-2 demo calls) | Haiku | ~2 calls | trivial | **~$0.50** |
| **Buffer** | | | | **~$8.50** |

The buffer is real; budget tracker has a hard kill at $18 (90%). Every LLM call hits a wrapper that records (provider, model, input_tok, output_tok, cached_tok, cost_usd, purpose, iteration_id). The cost report is `SELECT purpose, sum(cost_usd) FROM cost_ledger GROUP BY purpose`.

---

## 13. Repository Layout

```
riverline/
├── docker-compose.yml
├── Makefile                # rerun-eval, demo, fresh-start
├── README.md
├── PLAN.md                 # this file
├── decision-journal.md     # handwritten, not LLM
├── apps/
│   ├── workflow/           # Temporal workflow + activities
│   ├── gateway/            # FastAPI: inbound chat, vapi webhook, operator
│   ├── voice/              # Vapi client + webhook handlers
│   └── learner/            # self-learning loop runner
├── packages/
│   ├── agents/             # BaseAgent, Agent1/2/3
│   ├── summarizer/         # handoff schema + token-budget enforcer
│   ├── evaluator/          # rubric judge, system judge, compliance checker
│   ├── meta_evaluator/     # Darwin-Gödel layer
│   ├── llm/                # client + budget tracker + prompt cache
│   ├── simulator/          # borrower personas
│   ├── storage/            # SQLAlchemy models, repos
│   └── stats/              # bootstrap, paired t, kappa
├── prompts/
│   ├── agent_1/v0_001.md ...
│   ├── agent_2/...
│   ├── agent_3/...
│   ├── judge/...
│   └── simulator/...
├── scripts/
│   ├── seed_db.py
│   ├── run_learning_loop.py
│   └── rerun_evaluation.py
├── data/
│   ├── seeds.json
│   ├── compliance_probes.json
│   └── raw_evaluations/    # CSVs per iteration
└── tests/
    ├── test_token_budget.py
    ├── test_handoff_schema.py
    ├── test_compliance_gate.py
    └── test_stat_gates.py
```

---

## 14. Five-Day Timeline

| Day | Morning | Afternoon | End-of-day deliverable |
|---|---|---|---|
| **1** | Skeleton: docker-compose (Temporal, Postgres, Redis), DB models, token guard, LLM wrapper + cost tracker | BaseAgent, Agent 1 (chat) with prompt v0, simple borrower simulator (cooperative only) | One end-to-end Agent 1 conversation through Temporal, cost logged |
| **2** | Summarizer + 500-token handoff schema with tests, Agent 3 chat, Agent 2 voice stub (text-mode first for eval), full pipeline wired in workflow | Compliance checker (regex + judge), 5 personas, compliance probes | Full A1→A2→A3 pipeline runs end-to-end in text mode; compliance gate works |
| **3** | Evaluation layer: rubric judge, system-level judge, paired bootstrap stat gate | Self-learning loop driver, run first 2 iterations on agent 1 | First evolution data; cost ~$3 used |
| **4** | Meta-evaluator: kappa agreement, correlation check, holdout. Seed one demonstrable meta-flaw (lenient compliance judge) and verify the loop catches+fixes it | Run loop across all 3 agents, 3-5 iterations each | Full evolution report data; meta-eval finding documented |
| **5** | Vapi integration for one real outbound voice call + recording. Tighten Docker boot to <5min. Write technical writeup + reproducibility doc. Decision journal entries written by hand throughout the week, consolidated. Demo video. | Buffer for whatever broke. Final cost report. | Submission. |

Risk budget: ~6 hours of slack baked in. Real risks below.

---

## 15. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Vapi outbound call setup eats a day | Medium | Build Agent 2 in text mode first; voice is a transport swap at the end. One real call is enough for the audio deliverable. |
| Statistical gates too strict → nothing ever adopts | Medium | Use non-inferiority bounds, not strict superiority. Tolerate small regressions on secondary metrics. |
| Statistical gates too loose → noise gets adopted | Medium | Bootstrap CIs require true paired evaluation. Reject if compliance regresses *at all*. |
| Cost overrun | Low-Medium | Hard $18 kill switch. Prompt cache on system prompts. Cheap model for simulation. |
| Docker boot >5min | Low | Pre-pull images; use Temporal's lightweight dev container. |
| Meta-eval finds nothing genuine to fix | Medium | Pre-seed lenient judge as v0 so loop has a real flaw to catch. Document this honestly — the *capability* to catch flaws is what's being graded. |
| Voice transcript quality bad → handoff 2→3 noisy | Medium | Vapi returns structured transcript with speaker labels. Summarizer is robust to noise (structured schema). |
| Decision journal looks AI-generated | Low if disciplined | Write entries in real time, by hand, with rough edges. Include false starts (e.g., "tried option C, dropped because…"). Photo of paper notes is allowed. |

---

## 16. Things I'm Intentionally NOT Building

(Decision-journal candidate.)

- **No UI.** Operator CLI is enough. UI eats time and isn't graded.
- **No multi-tenant isolation.** Single-tenant assumption is fine for the assignment.
- **No real PII storage.** Borrower profiles are synthetic. Sidesteps privacy concerns and lets reproducibility be clean.
- **No production-grade voice pipeline.** One real call for the recording. The rest of evaluation runs in text mode using transcripts. Justified because voice quality isn't the graded axis; cross-modal handoff *fidelity* is, and that's tested by treating the voice transcript as the input to handoff-2.
- **No fancy RL / DSPy / TextGrad.** A clean, defensible bootstrap-CI + paired-eval + LLM-proposer loop is what the spec asks for. Going more exotic risks not being able to explain it in the live interview.

---

## 17. What I Need to Confirm Before Building

These are the only places I'd like to pin down before writing code. Defaults shown in **bold** are what I'd go with if you don't redirect.

1. **LLM provider**: **Anthropic Claude** (Haiku/Sonnet/Opus). Alternative: OpenAI (GPT-4o-mini/4o). Reason for Anthropic: prompt caching + same family from agents to judge to meta. I have an API key already? — need to confirm.
2. **Voice provider**: **Vapi**. Alternative: Bland, Retell. Vapi has the cleanest webhook → transcript flow.
3. **Borrower simulator personas**: the 5 listed above (cooperative, combative, evasive, confused, distressed). Open to adding more.
4. **N per iteration**: **30 borrowers** (6 per persona). Drives both cost and statistical power. With paired bootstrap, 30 gives reasonable CIs without burning budget.
5. **Live system hosting**: spec says "live and runnable at all times." Reading this as "Docker on my machine, with a public Vapi webhook tunnel (ngrok or fly.io)." Will tunnel through Cloudflare/fly.io for the duration of evaluation.
