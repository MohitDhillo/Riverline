"""Make one real outbound Vapi call for the audio-deliverable.

This bypasses Temporal for simplicity — Day 5 deliverable is just the recording.
Steps:
  1. Pick a seeded borrower (or any phone number you pass)
  2. Build a representative handoff payload from Agent 1's stub (or run a real
     A1 conversation first if you want the chat→voice seam to be authentic)
  3. POST to Vapi /call/phone via VapiClient
  4. Wait for the call to complete (poll Vapi every 5s)
  5. Print the recordingUrl when the call ends

Prerequisites in .env:
  ANTHROPIC_API_KEY   = sk-ant-...      (used by Vapi's Anthropic provider)
  VAPI_API_KEY        = ...             (https://dashboard.vapi.ai/)
  VAPI_PHONE_NUMBER_ID= ...             (provision a number in Vapi dashboard)
  DEMO_BORROWER_PHONE = +14155551234    (your own number for the demo)
  PUBLIC_WEBHOOK_URL  = https://...     (Cloudflare Tunnel pointed at FastAPI :8000)

Run:
  cloudflared tunnel --url http://localhost:8000     # in another terminal
  uv run uvicorn apps.gateway.main:app --port 8000   # in another terminal
  uv run python scripts/vapi_call.py                 # places the call
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.learner.loop import _STUB_HANDOFFS_TO_AGENT_2
from apps.voice.client import VapiClient
from packages.config import settings
from packages.simulator.borrower import load_borrowers


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--persona", default="cooperative",
                   choices=["cooperative", "distressed", "combative"])
    p.add_argument("--to", default=None,
                   help="Phone number to call. Defaults to DEMO_BORROWER_PHONE in .env")
    p.add_argument("--borrower-id", default=None,
                   help="Optional borrower UUID prefix to grab a name from seeds.")
    p.add_argument("--poll-seconds", type=int, default=10)
    p.add_argument("--max-wait", type=int, default=600)
    args = p.parse_args()

    s = settings()
    target = args.to or s.demo_borrower_phone
    if not target:
        print("ERROR: --to or DEMO_BORROWER_PHONE in .env required"); return 2

    # Pick a borrower name for the metadata (optional)
    borrower_name = "borrower"
    if args.borrower_id:
        bs = [b for b in load_borrowers() if b.id.startswith(args.borrower_id)]
        if bs:
            borrower_name = bs[0].name

    handoff = _STUB_HANDOFFS_TO_AGENT_2[args.persona]
    print(f"Placing Vapi outbound call:")
    print(f"  to       : {target}")
    print(f"  persona  : {args.persona}")
    print(f"  borrower : {borrower_name}")
    print(f"  webhook  : {s.public_webhook_url or '(not configured — recording will be in Vapi dashboard)'}")
    print()

    client = VapiClient()
    res = client.start_outbound_call(
        to_number=target,
        handoff_json=handoff,
        borrower_name=borrower_name,
    )
    print(f"Call initiated. call_id={res.call_id}  status={res.status}")
    if not res.call_id:
        print("ERROR: Vapi did not return a call_id. Raw response:")
        print(json.dumps(res.raw, indent=2))
        return 1

    waited = 0
    while waited < args.max_wait:
        time.sleep(args.poll_seconds)
        waited += args.poll_seconds
        info = client.get_call(res.call_id)
        status = info.get("status", "unknown")
        ended = info.get("endedAt") or status in ("ended", "no-answer", "busy", "failed")
        print(f"  [{waited:>4}s] status={status}")
        if ended:
            print()
            print("=" * 60)
            print(f"Call ended. status={status}")
            recording = info.get("recordingUrl") or info.get("artifact", {}).get("recordingUrl")
            print(f"recordingUrl: {recording or '(not yet available — check dashboard in 30s)'}")
            print(f"endedReason : {info.get('endedReason')}")
            if info.get("transcript"):
                print(f"transcript  : {info['transcript'][:400]}")
            return 0

    print(f"Timed out after {args.max_wait}s. Check dashboard: https://dashboard.vapi.ai/calls/{res.call_id}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
