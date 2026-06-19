from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging

from fastmcp import Client
from icd10_server import mcp

logger = logging.getLogger(__name__)


def _run_in_thread(coro) -> any:
    """Run an async coroutine from Streamlit's synchronous context.

    Streamlit 1.40+ already has a running event loop in the main thread,
    so asyncio.run() raises RuntimeError there. Spawning a fresh thread gives
    a clean loop-free context for asyncio.run().
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result(timeout=60)


async def _search_async(query: str, max_results: int) -> list[dict]:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "search_icd10_codes",
            {"query": query, "max_results": max_results},
        )
        # FastMCP 3.x: result.data is the already-deserialized return value
        if result.data is not None:
            return result.data if isinstance(result.data, list) else []
        # Fallback: parse from first TextContent
        if result.content:
            raw = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
            return json.loads(raw)
        return []


async def _describe_async(code: str) -> dict:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_icd10_description",
            {"code": code},
        )
        if result.data is not None:
            return result.data if isinstance(result.data, dict) else {}
        if result.content:
            raw = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
            return json.loads(raw)
        return {}


def search_icd10_codes(query: str, max_results: int = 10) -> list[dict]:
    """Sync entry point: search ICD-10-CM codes by clinical term."""
    try:
        return _run_in_thread(_search_async(query, max_results))
    except Exception as e:
        logger.error("ICD-10 search failed for '%s': %s", query, e)
        return []


def get_icd10_description(code: str) -> dict:
    """Sync entry point: get details for a specific ICD-10-CM code."""
    try:
        return _run_in_thread(_describe_async(code))
    except Exception as e:
        logger.error("ICD-10 description lookup failed for '%s': %s", code, e)
        return {"error": str(e), "code": code}
