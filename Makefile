.PHONY: fresh-start up down seed test smoke worker api logs clean

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
