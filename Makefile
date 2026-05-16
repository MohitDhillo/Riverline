.PHONY: fresh-start up down seed test smoke chat chat-sim vapi-call rerun-eval meta-eval lift-prompts report worker api logs clean costs

# Boot everything from cold (target: <5 min, FINAL_PLAN §13)
fresh-start: up seed test
	@echo "✓ fresh-start complete. run 'make smoke' for an end-to-end agent run."

up:
	docker compose up -d postgres redis temporal
	@echo "waiting for temporal to be healthy..."
	@until docker compose ps temporal --format json | grep -q '"Health":"healthy"'; do sleep 2; done
	@echo "✓ infrastructure up"

down:
	docker compose down

seed:
	uv run python scripts/seed_db.py

test:
	uv run pytest tests/ -v

smoke:
	uv run python scripts/smoke_test.py

# Interactive: YOU play the borrower. Walks through A1 -> A2 -> A3.
#   make chat                    -> cooperative profile
#   make chat PERSONA=distressed -> distressed profile (tests hardship rule)
chat:
	uv run python scripts/chat.py --mode human $(if $(PERSONA),--persona $(PERSONA),)

# Autoplay: LLM-borrower against the agents (visible alternative to make smoke).
chat-sim:
	uv run python scripts/chat.py --mode sim $(if $(PERSONA),--persona $(PERSONA),)

vapi-call:
	uv run python scripts/vapi_call.py $(if $(PERSONA),--persona $(PERSONA),) $(if $(TO),--to $(TO),)

rerun-eval:
	uv run python scripts/run_learning_loop.py --agent $(or $(AGENT),agent_1) --iters $(or $(ITERS),2) --n $(or $(N),15) --variants $(or $(VARIANTS),2) --eval-mode $(or $(EVAL_MODE),full)

meta-eval:
	uv run python scripts/run_meta_eval.py $(if $(ITERATION),--iteration $(ITERATION),)

# Write currently-active prompts back to disk so adoptions are visible in the repo.
lift-prompts:
	uv run python scripts/lift_active_prompts.py

# Build the consolidated EVOLUTION_REPORT.md (pure DB + CSV aggregation, no LLM).
report:
	uv run python scripts/build_evolution_report.py

worker:
	uv run python -m apps.workflow.worker

api:
	uv run uvicorn apps.gateway.main:app --reload --host 0.0.0.0 --port 8000

logs:
	docker compose logs -f --tail=50

clean:
	docker compose down -v
	rm -rf .venv __pycache__ .pytest_cache

costs:
	uv run python -c "from packages.storage.db import session_scope; from packages.storage.models import CostLedgerEntry; from sqlalchemy import select, func; \
	  s = session_scope().__enter__(); \
	  rows = s.execute(select(CostLedgerEntry.purpose, func.count(), func.sum(CostLedgerEntry.cost_usd)).group_by(CostLedgerEntry.purpose)).all(); \
	  total = 0.0; \
	  [print(f'{r[0]:25s} calls={r[1]:4d}  USD={float(r[2] or 0):.6f}') or (total := total + float(r[2] or 0)) for r in rows]; \
	  print(f'TOTAL = USD {total:.6f}')"
