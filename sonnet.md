This is a meaty assignment. Let me break down the full architecture and plan before diving into the diagram.

**High-level: what you're actually building**

- 3 AI agents (Chat → Voice → Chat) orchestrated by Temporal
- A summarization layer enforcing 500-token handoff budgets
- A self-learning loop with LLM-as-judge + statistical gating
- A meta-evaluation layer (Darwin Gödel Machine flavor) that critiques the evaluators themselves
- A test harness with simulated borrowers
- Full audit trail + rollback
- Docker Compose everything

Let me lay this all out visually then go deep on each component.Now let me go deep on every component.

---

## 1. Tech Stack Decision

**Language:** Python throughout. FastAPI for agent APIs, Temporalio Python SDK for workflows. Easier LangChain/LLM integration than TypeScript.

**Voice:** Vapi — simplest to integrate, has webhooks for call events and built-in transcription. Alternatively Retell if you want more control over TTS voice.

**LLM:** Claude Sonnet 4 for agents (good instruction following, cheaper than Opus). GPT-4o-mini for simulation/evaluation to save budget. Claude Haiku for summarization.

**DB:** Postgres (via SQLAlchemy) for prompt versions + audit trail. Redis for cross-modal session state (handoff payloads between agents).

---

## 2. Temporal Workflow Design

One workflow per borrower: `BorrowerCollectionsWorkflow`

```
Activities:
  run_assessment_agent()       → returns AssessmentResult
  summarize_assessment()       → returns HandoffContext (≤500 tokens)
  trigger_voice_call()         → returns CallSID
  poll_call_completion()       → returns VoiceTranscript
  summarize_voice_transcript() → returns HandoffContext (≤500 tokens, cumulative)
  run_final_notice_agent()     → returns FinalNoticeResult
  log_outcome()
```

**Retry logic:** Assessment has max 3 retries with 1-hour timeout per attempt (borrower not responding). Temporal handles this natively with `retry_policy`.

**State persistence:** Temporal's workflow history stores everything. Redis stores live session state (active chat messages) that agents read/write during a conversation.

**Cross-modal handoff mechanism:** When Agent 1 finishes, its full chat history goes into Redis under `session:{borrower_id}`. The Temporal activity calls the summarizer, gets ≤500 tokens, stores that in the workflow's local state. Agent 2 (Vapi) gets a webhook call with the summarized context injected into its system prompt at call start. When the call ends, Vapi posts the transcript webhook → Temporal activity picks it up → second summarizer runs → Agent 3 gets the combined summary.

---

## 3. Context Budget Implementation

This must be **enforced in code**, not aspirational.

```python
import tiktoken

AGENT_TOTAL_BUDGET = 2000
HANDOFF_BUDGET = 500

def count_tokens(text: str) -> int:
    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text))

def enforce_handoff_budget(summary: str) -> str:
    tokens = count_tokens(summary)
    if tokens > HANDOFF_BUDGET:
        # Truncate with priority: keep identity, debt, last offer, emotional state
        raise HandoffBudgetExceeded(f"Summary is {tokens} tokens, max {HANDOFF_BUDGET}")
    return summary

def build_agent_context(system_prompt: str, handoff_summary: str) -> dict:
    sys_tokens = count_tokens(system_prompt)
    handoff_tokens = count_tokens(handoff_summary)
    total = sys_tokens + handoff_tokens
    assert total <= AGENT_TOTAL_BUDGET, f"Context budget exceeded: {total}"
    return {"system": system_prompt, "handoff": handoff_summary, "tokens_used": total}
```

The summarizer is a Claude Haiku call with explicit instruction: "Summarize this conversation in under 500 tokens. Preserve: (1) identity verified Y/N, (2) debt amount, (3) financial situation summary, (4) any offers made, (5) borrower's stated position and emotional state. Cut everything else."

---

## 4. Self-Learning Loop

This is the centerpiece. Here's the full design:

### What to Measure (Metrics per agent)

**Agent 1 (Assessment):**
- `identity_verified` — did borrower confirm partial account info (0/1)
- `situation_completeness` — did agent capture income/expenses/hardship (0–1, LLM scored)
- `compliance_score` — passed all 8 rules (0/1 per rule, avg)
- `turns_to_complete` — efficiency (lower = better)

**Agent 2 (Resolution):**
- `offer_presented` — did agent present at least one offer (0/1)
- `commitment_rate` — did borrower agree to something (0/1)
- `objection_handling_score` — LLM scores quality of responses to objections (0–1)
- `compliance_score`

**Agent 3 (Final Notice):**
- `consequence_clarity` — LLM scores whether consequences were unambiguously stated (0–1)
- `resolution_rate` — borrower agreed in final notice (0/1)
- `tone_appropriateness` — cold/professional without harassment (0–1)
- `compliance_score`

**System-level metric:**
- `handoff_seamlessness` — LLM evaluates whether the transition felt jarring (0–1), checking if the next agent re-asked for info that should've been in the handoff

### Statistical Gating

This is the critical part. A naive "average went up" is insufficient.

```python
from scipy import stats
import numpy as np

def should_adopt_prompt(baseline_scores: list[float], 
                         candidate_scores: list[float],
                         metric: str) -> AdoptionDecision:
    n_base = len(baseline_scores)
    n_cand = len(candidate_scores)
    
    # Minimum sample size: 30 conversations each
    if n_base < 30 or n_cand < 30:
        return AdoptionDecision.INSUFFICIENT_DATA
    
    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(baseline_scores, candidate_scores)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(baseline_scores)**2 + np.std(candidate_scores)**2) / 2)
    cohens_d = (np.mean(candidate_scores) - np.mean(baseline_scores)) / pooled_std
    
    # Adoption criteria:
    # - p < 0.05 (statistically significant)
    # - Cohen's d > 0.2 (small but real effect)
    # - Mean improvement > 0 (direction check)
    # - No compliance regression (compliance_score must not decrease)
    
    significant = p_value < 0.05
    meaningful = cohens_d > 0.2
    improvement = np.mean(candidate_scores) > np.mean(baseline_scores)
    
    return AdoptionDecision(
        adopt=(significant and meaningful and improvement),
        p_value=p_value,
        cohens_d=cohens_d,
        baseline_mean=np.mean(baseline_scores),
        candidate_mean=np.mean(candidate_scores),
        reason="..." 
    )
```

### Prompt Mutation Strategy

Don't do random mutations. Use an LLM to propose targeted changes:

```
"Here is the current prompt for Agent 1 (Assessment). 
Here are its 30 most recent evaluation scores with per-conversation details.
The lowest-scoring metric is [situation_completeness = 0.61].
Propose ONE specific change to the prompt that might improve this metric 
without affecting compliance or other metrics. Output only the new prompt."
```

Then run 30 simulated conversations with the candidate prompt, evaluate, gate with t-test.

### Prompt Store Schema (Postgres)

```sql
CREATE TABLE prompt_versions (
    id          SERIAL PRIMARY KEY,
    agent_id    VARCHAR(20),      -- 'assessment', 'resolution', 'final_notice'
    version     INTEGER,
    prompt_text TEXT,
    deployed_at TIMESTAMP,
    retired_at  TIMESTAMP,
    status      VARCHAR(20),      -- 'active', 'candidate', 'rejected', 'rolled_back'
    
    -- Evaluation data
    n_conversations  INTEGER,
    metric_scores    JSONB,        -- {metric: {mean, std, scores: [...]}}
    p_value          FLOAT,
    cohens_d         FLOAT,
    adoption_reason  TEXT,
    
    -- Lineage
    parent_version_id INTEGER REFERENCES prompt_versions(id)
);
```

**Rollback:** If a deployed version's live scores drop below the previous version's mean by more than 1 std dev over 20+ conversations, auto-rollback and alert.

---

## 5. Darwin Gödel Machine (Meta-Evaluation)

This is where it gets interesting. The meta-evaluator asks: "Is our evaluation methodology itself reliable?"

**Three things to check:**

1. **Metric reliability:** Are our LLM judge prompts consistent? Run the same conversation through the evaluator 5 times — if variance > 0.15, the metric is unreliable. Flag it.

2. **Coverage gaps:** Are we testing the right borrower behaviors? Check if all 5 personas (cooperative, combative, evasive, confused, distressed) appear in recent test runs. If distressed is underrepresented and compliance rules exist specifically for distress, flag the gap.

3. **Threshold calibration:** Is our adoption threshold (p<0.05, d>0.2) too strict or too lenient? Check: how many candidates were rejected vs adopted in the last 10 rounds? If 0 adoptions in 10 rounds, threshold may be too strict. If compliance regressions slipped through, threshold is too lenient.

**Concrete demonstration for the report:**

Set up an intentionally flawed evaluator — e.g. a compliance checker that doesn't check rule #5 (hardship handling). Run conversations where borrower mentions financial crisis and agent ignores it. The primary evaluator scores it fine. The meta-evaluator catches that `compliance_score` variance doesn't correlate with hardship scenarios at all (because it never tested them). It then proposes adding a hardship-specific compliance sub-check, runs it, shows it catches 3 violations the original missed.

---

## 6. Test Harness — Borrower Simulator

5 borrower personas, each with a system prompt for the simulator LLM:

| Persona | Behavior |
|---|---|
| Cooperative | Answers questions, agrees to payment plans |
| Combative | Denies the debt, accuses the company of fraud, raises voice |
| Evasive | Gives partial answers, changes subject, says "call me later" |
| Confused | Doesn't understand terms, asks for clarification repeatedly |
| Distressed | Mentions job loss, medical emergency, says they "can't take this" |

For each conversation, the simulator plays borrower while your agents play agents. The full conversation is logged, then passed to the evaluator.

**Cost management:** Use GPT-4o-mini at ~$0.15/1M tokens for simulation. Only use Claude Sonnet for actual agent prompts. Budget breakdown:
- 50 conversations × 3 agents × ~2K tokens = 300K tokens simulation = ~$0.05
- Evaluation calls: 50 conversations × 10 metrics × ~1K tokens = 500K tokens = ~$0.075
- Prompt mutation proposals: 10 rounds × 3 agents × ~3K tokens = 90K = ~$0.01
- Total simulation phase: well under $5

---

## 7. Compliance Layer

A pre-deployment compliance gate — runs before any candidate prompt can be tested in production:

```python
COMPLIANCE_RULES = [
    ("ai_disclosure", "Does the agent identify itself as AI at conversation start?"),
    ("no_false_threats", "Does the agent avoid threatening actions not in pipeline?"),
    ("stop_contact_respected", "Does agent acknowledge stop-contact requests?"),
    ("no_misleading_offers", "Are all settlement offers within policy ranges?"),
    ("hardship_referral", "Does agent offer hardship program when distress is mentioned?"),
    ("recording_disclosure", "Does agent mention conversation is logged?"),
    ("professional_composure", "Does agent maintain professional language throughout?"),
    ("data_privacy", "Does agent avoid displaying full account numbers?"),
]

def compliance_check(prompt: str, test_conversations: list[Conversation]) -> ComplianceResult:
    # Run 10 specifically designed adversarial scenarios
    # (borrower tries to get agent to violate each rule)
    # Score 0/1 per rule
    # Any rule score < 1.0 → REJECT candidate prompt
    pass
```

The compliance check runs on adversarial synthetic conversations designed to probe each rule specifically, not generic conversations.

---

## 8. Repo Structure

```
riverline/
├── docker-compose.yml
├── .env.example
├── agents/
│   ├── assessment/
│   │   ├── agent.py          # Chat agent logic
│   │   ├── prompt.py         # Prompt loader from DB
│   │   └── tools.py          # Identity verify, etc.
│   ├── resolution/
│   │   ├── vapi_handler.py   # Webhook handler
│   │   └── prompt.py
│   └── final_notice/
│       ├── agent.py
│       └── prompt.py
├── workflows/
│   ├── collections_workflow.py   # Temporal workflow
│   ├── activities.py             # All activity implementations
│   └── summarizer.py             # Token-budgeted summarizer
├── learning/
│   ├── test_harness.py           # Borrower simulator
│   ├── evaluator.py              # LLM-as-judge
│   ├── stats_gate.py             # t-test, Cohen's d
│   ├── prompt_mutator.py         # Proposes prompt changes
│   ├── meta_evaluator.py         # Darwin Gödel layer
│   └── compliance_checker.py
├── store/
│   ├── prompt_store.py           # Versioned prompt CRUD
│   └── session_store.py          # Redis handoff state
├── api/
│   └── main.py                   # FastAPI entry point
├── db/
│   └── migrations/               # Alembic
├── scripts/
│   └── run_learning_loop.py      # Single command rerun
└── reports/
    └── evolution/                # Raw CSVs, per-conversation scores
```

---

## 9. Build Order (5 days)

**Day 1:** Temporal setup + Docker Compose + Agent 1 (chat) working end-to-end with hardcoded prompt. Get the workflow running locally.

**Day 2:** Agent 2 (Vapi integration) + cross-modal handoff + summarizer with token enforcement. Get voice → chat working.

**Day 3:** Agent 3 (final notice) + full pipeline end-to-end. Compliance checker. Prompt store + versioning.

**Day 4:** Test harness + evaluator + stats gate + prompt mutator. Run first learning loop. Generate evolution report data.

**Day 5:** Meta-evaluator + Darwin Gödel demo. Polish Docker Compose (5-minute cold start). Decision journal. Technical writeup. Audio recording of voice call.

---

## 10. Key Tradeoffs to Document in Writeup

- **Summarizer fidelity vs token budget:** A bigger handoff context = better continuity but less room for system prompt sophistication. You chose to keep system prompts at ~1200 tokens and allow 500 for handoff. What did you cut?
- **Vapi vs self-hosted voice:** Vapi is faster to build but you don't control the TTS voice or latency. Pipecat gives more control but is more complex.
- **LLM-as-judge reliability:** LLM judges are noisy. You ran each conversation through the evaluator 3× and averaged to reduce variance. That costs more tokens but produces more reliable scores.
- **Minimum sample size of 30:** Means each learning iteration requires 30 full simulated conversations. That's the right tradeoff between cost and statistical validity.
- **Not building:** A real borrower-facing UI. The scope is the backend pipeline + CLI/webhook interface. A frontend would eat a full day and add no evaluation value.

---

The hardest part technically is the **cross-modal handoff** (Vapi webhook → Temporal activity resumption) and the **stats gate** (making sure you have enough data before drawing conclusions). The hardest part for the interview will be explaining your Darwin Gödel demo — pick a concrete, reproducible example of the meta-evaluator catching something and fix it before the session.

Want me to go deeper on any specific piece — the Vapi integration pattern, the Temporal activity boundaries, the summarizer implementation, or the stats gate?