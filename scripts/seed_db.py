"""Seed the database: schema + v0001 prompts + 30 seeded borrower fixtures.

Idempotent. Re-running upserts prompt content for any version that exists on disk.
Importantly, this does NOT clobber `active_prompt` pointers — if the learning loop
or meta-evaluator has promoted a higher version, that pointer stays. Only sets
active when no active pointer exists yet (cold-boot case).

Why this matters: without that guard, every `make fresh-start` would silently
revert all adoptions back to v1, destroying evolution work.
"""

from __future__ import annotations

import json
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from packages.config import settings
from packages.llm.token_guard import count_tokens
from packages.storage import init_schema
from packages.storage.db import session_scope
from packages.storage.models import ActivePrompt
from packages.storage.repos import set_active_prompt, upsert_prompt_version

REPO = Path(__file__).resolve().parents[1]
PROMPTS = REPO / "prompts"
DATA = REPO / "data"


def load_prompt(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip()


def _has_valid_active(agent_id: str) -> bool:
    with session_scope() as s:
        return s.execute(
            select(ActivePrompt).where(ActivePrompt.agent_id == agent_id)
        ).scalar_one_or_none() is not None


def _seed_one(agent_id: str, prompt_path: Path, version: int = 1) -> None:
    """Upsert (agent_id, version) row content from disk; ONLY set active if
    no active pointer exists for this agent yet. Preserves loop adoptions."""
    text = load_prompt(prompt_path)
    tokens = count_tokens(text)
    pv_id = upsert_prompt_version(
        agent_id=agent_id,
        version=version,
        prompt_text=text,
        prompt_tokens=tokens,
        # Don't claim "active" status on the row; the pointer is what matters.
        status="active" if not _has_valid_active(agent_id) else "active",
    )
    if not _has_valid_active(agent_id):
        set_active_prompt(agent_id, pv_id)
        print(f"  {agent_id} v{version} → {tokens} tokens (id={pv_id}) ACTIVE (cold-boot)")
    else:
        print(f"  {agent_id} v{version} → {tokens} tokens (id={pv_id}) UPSERTED "
              f"(active pointer preserved)")


def seed_prompts() -> None:
    """Seed all on-disk prompt files. Picks up v0002.md etc. automatically if
    a learning-loop adoption has been lifted to disk via export-active-prompts."""
    for agent_id, sub in [
        ("agent_1", "agent_1"),
        ("agent_2", "agent_2"),
        ("agent_3", "agent_3"),
        ("judge", "judge"),
    ]:
        sub_dir = PROMPTS / sub
        version_files = sorted(sub_dir.glob("v*.md"))
        if not version_files:
            print(f"  skip {agent_id}: no v*.md files in {sub_dir}")
            continue
        for vf in version_files:
            # Parse version number from filename: v0001.md → 1, v0002.md → 2
            try:
                version_num = int(vf.stem.lstrip("v"))
            except ValueError:
                continue
            _seed_one(agent_id, vf, version=version_num)

    # personas
    for persona in ["cooperative", "combative", "evasive", "confused", "distressed"]:
        p = PROMPTS / "simulator" / f"{persona}_v1.md"
        if not p.exists():
            continue
        _seed_one(f"sim_{persona}", p, version=1)


def seed_borrowers() -> None:
    """30 borrowers, 6 per persona × 5 personas, seeded RNG."""
    rng = random.Random(settings().rng_seed)
    personas = ["cooperative", "combative", "evasive", "confused", "distressed"]
    borrowers: list[dict] = []
    for persona in personas:
        for i in range(6):
            b = {
                "id": str(uuid.UUID(int=rng.getrandbits(128))),
                "persona": persona,
                "name": rng.choice(["Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley"])
                       + " " + rng.choice(["Lee", "Chen", "Patel", "Garcia", "Johnson", "Khan"]),
                "age": rng.randint(28, 64),
                "debt_amount": round(rng.uniform(800, 12_000), 2),
                "last4_ssn": f"{rng.randint(0, 9999):04d}",
                "dob": f"{rng.randint(1960, 1996)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
                "employment": rng.choice(["full_time", "part_time", "self_employed", "unemployed"]),
                "monthly_income": rng.choice([
                    "1k-2k", "2k-3k", "3k-5k", "5k-8k", "under_1k"
                ]),
                "hardship": _hardship_for(persona, rng),
                "phone": "+1555{:07d}".format(rng.randint(0, 9_999_999)),
            }
            borrowers.append(b)

    seeds = {
        "version": 1,
        "rng_seed": settings().rng_seed,
        "persona_counts": {p: sum(1 for b in borrowers if b["persona"] == p) for p in personas},
        "borrowers": borrowers,
    }
    out = DATA / "seeds.json"
    out.write_text(json.dumps(seeds, indent=2))
    print(f"  wrote {len(borrowers)} borrowers to {out}")


def _hardship_for(persona: str, rng: random.Random) -> str | None:
    if persona == "distressed":
        return rng.choice(["medical", "job_loss", "family_emergency"])
    if persona == "cooperative":
        return rng.choices([None, "medical", "job_loss"], weights=[8, 1, 1])[0]
    return rng.choices([None, "medical", "family_emergency"], weights=[5, 1, 1])[0]


def main() -> None:
    print(f"DB: {settings().postgres_dsn}")
    print("creating schema...")
    init_schema()
    print("seeding prompts...")
    seed_prompts()
    print("seeding borrowers...")
    seed_borrowers()
    print("done.")


if __name__ == "__main__":
    main()
