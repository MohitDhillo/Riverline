"""Temporal worker — registers workflows and activities, listens on the task queue."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker

from apps.workflow.activities import run_chat_agent
from apps.workflow.collections import CollectionsWorkflow
from packages.config import settings

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    s = settings()
    client = await Client.connect(s.temporal_host, namespace=s.temporal_namespace)
    logging.info(f"connected to temporal at {s.temporal_host}")
    # Activities are sync (they call sync sqlalchemy + sync anthropic SDK).
    # Run them on a thread pool so the worker event loop isn't blocked.
    activity_executor = ThreadPoolExecutor(max_workers=8)
    async with Worker(
        client,
        task_queue=s.temporal_task_queue,
        workflows=[CollectionsWorkflow],
        activities=[run_chat_agent],
        activity_executor=activity_executor,
    ):
        logging.info(f"worker listening on task queue '{s.temporal_task_queue}'")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
