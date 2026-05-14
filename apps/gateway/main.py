"""FastAPI gateway.

Day 1: minimum surface area — start a workflow, query its outcome.
Day 2 adds: send-message (signal into chat agent), Vapi webhook receiver.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from temporalio.client import Client

from apps.voice.webhook import router as voice_router
from apps.workflow.collections import CollectionsInput, CollectionsWorkflow
from packages.config import settings

app = FastAPI(title="Riverline Collections Gateway")
app.include_router(voice_router)
_client: Client | None = None


async def temporal() -> Client:
    global _client
    if _client is None:
        s = settings()
        _client = await Client.connect(s.temporal_host, namespace=s.temporal_namespace)
    return _client


class StartWorkflowRequest(BaseModel):
    borrower_id: str
    iteration_id: int | None = None


class StartWorkflowResponse(BaseModel):
    workflow_id: str
    run_id: str


@app.post("/workflows/start", response_model=StartWorkflowResponse)
async def start_workflow(req: StartWorkflowRequest) -> StartWorkflowResponse:
    s = settings()
    c = await temporal()
    workflow_id = f"collections-{uuid.uuid4().hex[:8]}"
    handle = await c.start_workflow(
        CollectionsWorkflow.run,
        CollectionsInput(borrower_id=req.borrower_id, iteration_id=req.iteration_id),
        id=workflow_id,
        task_queue=s.temporal_task_queue,
    )
    return StartWorkflowResponse(workflow_id=workflow_id, run_id=handle.result_run_id or "")


@app.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    c = await temporal()
    try:
        h = c.get_workflow_handle(workflow_id)
        desc = await h.describe()
        return {"status": desc.status.name, "workflow_id": workflow_id}
    except Exception as e:
        raise HTTPException(404, detail=str(e))


@app.get("/healthz")
async def healthz():
    return {"ok": True}
