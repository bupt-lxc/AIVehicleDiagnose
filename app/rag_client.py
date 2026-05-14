from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Any

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
except Exception:
    ClientSession = None
    sse_client = None

from .config import Settings
from .observability import AuditContext, AuditLogger


logger = logging.getLogger(__name__)


class RAGClient:
    def __init__(self, settings: Settings, audit_logger: AuditLogger | None = None) -> None:
        self._server_url = settings.rag_server_url
        self._sse_url = self._build_sse_url(settings.rag_server_url)
        self._retry_count = settings.mcp_retry_count
        self._retry_backoff_ms = settings.mcp_retry_backoff_ms
        self._audit_logger = audit_logger
        self._default_user_id = settings.user_id
        self._default_session_id = settings.session_id
        logger.info("RAG client initialized with standard MCP SDK SSE endpoint: %s", self._sse_url)

    def _build_sse_url(self, server_url: str) -> str:
        normalized_url = server_url.rstrip("/")
        if normalized_url.endswith("/sse"):
            return normalized_url
        return f"{normalized_url}/sse"

    async def _call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        trace_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
        turn_id: int = 0,
    ) -> dict[str, Any]:
        final_error = ""
        context = AuditContext(
            user_id=user_id or self._default_user_id,
            session_id=session_id or self._default_session_id,
            trace_id=trace_id,
            turn_id=turn_id,
            metadata={"tool_name": tool_name},
        )
        for attempt in range(self._retry_count + 1):
            start = time.perf_counter()
            try:
                if ClientSession is None or sse_client is None:
                    raise RuntimeError("mcp package is required for RAGClient tool calls")
                async with sse_client(self._sse_url) as streams:
                    read_stream, write_stream = streams
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        tool_result = await session.call_tool(
                            tool_name,
                            arguments=arguments,
                        )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                text_blocks = [
                    getattr(item, "text", "")
                    for item in getattr(tool_result, "content", [])
                    if getattr(item, "type", "") == "text"
                ]
                is_error = bool(getattr(tool_result, "isError", False))
                result = {
                    "content": "\n".join(block for block in text_blocks if block),
                    "citations": [],
                    "is_error": is_error,
                    "trace_id": trace_id,
                    "attempt": attempt + 1,
                }
                if self._audit_logger:
                    self._audit_logger.record_mcp_call(elapsed_ms, is_error=is_error)
                    self._audit_logger.emit(
                        "mcp_tool_call",
                        context,
                        {
                            "latency_ms": round(elapsed_ms, 3),
                            "attempt": attempt + 1,
                            "is_error": is_error,
                            "arguments": arguments,
                            "response_preview": result["content"][:800],
                        },
                    )
                if not is_error:
                    return result
                final_error = result["content"]
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                final_error = str(exc)
                if attempt < self._retry_count:
                    logger.warning("MCP tool call failed (%s) attempt=%s: %s", tool_name, attempt + 1, exc)
                else:
                    logger.exception("MCP tool call failed (%s) attempt=%s: %s", tool_name, attempt + 1, exc)
                if self._audit_logger:
                    self._audit_logger.record_mcp_call(elapsed_ms, is_error=True)
                    self._audit_logger.emit(
                        "mcp_tool_error",
                        context,
                        {
                            "latency_ms": round(elapsed_ms, 3),
                            "attempt": attempt + 1,
                            "error": str(exc),
                            "arguments": arguments,
                        },
                    )
            if attempt < self._retry_count:
                await asyncio.sleep(self._retry_backoff_ms / 1000.0)
        return {
            "content": f"MCP tool call failed ({tool_name}): {final_error}",
            "citations": [],
            "is_error": True,
            "trace_id": trace_id,
            "attempt": self._retry_count + 1,
        }

    async def query(
        self,
        question: str,
        top_k: int = 5,
        collection: str | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        min_score: float | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        turn_id: int = 0,
    ) -> dict[str, Any]:
        resolved_trace_id = trace_id or str(uuid.uuid4())
        arguments: dict[str, Any] = {"query": question, "top_k": top_k}
        if collection:
            arguments["collection"] = collection
        if dense_top_k is not None:
            arguments["dense_top_k"] = dense_top_k
        if sparse_top_k is not None:
            arguments["sparse_top_k"] = sparse_top_k
        if min_score is not None:
            arguments["min_score"] = min_score
        logger.info(
            "Executing single query via MCP SDK: top_k=%s collection=%s dense_top_k=%s sparse_top_k=%s min_score=%s",
            top_k,
            collection,
            dense_top_k,
            sparse_top_k,
            min_score,
        )
        return await self._call_tool(
            "query_knowledge_hub",
            arguments,
            trace_id=resolved_trace_id,
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
        )

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        turn_id: int = 0,
    ) -> dict[str, Any]:
        resolved_trace_id = trace_id or str(uuid.uuid4())
        return await self._call_tool(
            tool_name,
            arguments or {},
            trace_id=resolved_trace_id,
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
        )

    async def query_multi(
        self,
        queries: list[str],
        top_k: int = 5,
        collection: str | None = None,
        anchor_query: str | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        min_score: float | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        turn_id: int = 0,
    ) -> dict[str, Any]:
        resolved_trace_id = trace_id or str(uuid.uuid4())
        arguments: dict[str, Any] = {
            "multi_query": queries,
            "top_k": top_k,
        }
        if collection:
            arguments["collection"] = collection
        if anchor_query:
            arguments["query"] = anchor_query
        if dense_top_k is not None:
            arguments["dense_top_k"] = dense_top_k
        if sparse_top_k is not None:
            arguments["sparse_top_k"] = sparse_top_k
        if min_score is not None:
            arguments["min_score"] = min_score
        logger.info(
            "Executing multi query via MCP SDK: query_count=%s top_k=%s collection=%s dense_top_k=%s sparse_top_k=%s min_score=%s",
            len(queries),
            top_k,
            collection,
            dense_top_k,
            sparse_top_k,
            min_score,
        )
        return await self._call_tool(
            "query_knowledge_hub",
            arguments,
            trace_id=resolved_trace_id,
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
        )

    async def query_batch(
        self,
        questions: list[str],
        top_k: int = 5,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for question in questions:
            results.append(await self.query(question=question, top_k=top_k, collection=collection))
        return results

    def query_sync(self, question: str, top_k: int = 5, collection: str | None = None) -> dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
            
        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot call query_sync from a running event loop. "
                "Use 'await client.query(...)' instead."
            )
        
        return asyncio.run(self.query(question=question, top_k=top_k, collection=collection))

    async def list_collections(self) -> list[dict[str, Any]]:
        result = await self.call_tool("list_collections", {"include_stats": True})
        if result.get("is_error"):
            return []
        content = str(result.get("content", ""))
        collections: list[dict[str, Any]] = []
        for match in re.finditer(r"\*\*(?P<name>[^*]+)\*\*\s*-\s*(?P<count>\d+)\s+documents", content):
            collections.append(
                {
                    "name": match.group("name").strip(),
                    "count": int(match.group("count")),
                },
            )
        return collections
