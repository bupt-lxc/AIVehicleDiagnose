from __future__ import annotations

import importlib
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Callable
from collections import deque

from .rag_client import RAGClient


@dataclass(slots=True)
class ToolResult:
    name: str
    success: bool
    content: str
    latency_ms: float
    error: str = ""


@dataclass(slots=True)
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    handler: Callable[..., Any]
    input_schema: dict[str, Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._allowed_tools: set[str] | None = None
        self._plugin_load_errors: list[dict[str, str]] = []
        self._status_events = deque(maxlen=100)

    def _utcnow(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _record_status(self, event_type: str, tool_name: str, status: str, detail: str = "") -> None:
        self._status_events.append(
            {
                "timestamp": self._utcnow(),
                "event_type": event_type,
                "tool_name": tool_name,
                "status": status,
                "detail": detail,
            }
        )

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str = "",
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        self._tools[name] = ToolDefinition(
            name=name,
            description=description.strip(),
            handler=handler,
            input_schema=input_schema or {"type": "object", "properties": {}, "required": []},
        )
        self._record_status("load", name, "loaded")

    def unregister(self, name: str) -> bool:
        if name not in self._tools:
            return False
        self._tools.pop(name)
        self._record_status("unload", name, "unloaded")
        return True

    def load_plugins(self, plugin_paths: list[str]) -> None:
        for item in plugin_paths:
            try:
                module_name, attr = item.rsplit(":", 1)
                module = importlib.import_module(module_name)
                handler = getattr(module, attr)
                self.register(attr, handler)
            except Exception as exc:
                self._plugin_load_errors.append({"plugin": item, "error": str(exc)})
                self._record_status("load", item, "failed", str(exc))

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name].handler

    def set_allowed_tools(self, tool_names: list[str] | None) -> None:
        if tool_names is None:
            self._allowed_tools = None
            return
        normalized = {item.strip() for item in tool_names if item and item.strip()}
        self._allowed_tools = normalized if normalized else None

    def is_allowed(self, name: str) -> bool:
        if self._allowed_tools is None:
            return True
        return name in self._allowed_tools

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def describe_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": item.name, "description": item.description, "input_schema": item.input_schema}
            for item in sorted(self._tools.values(), key=lambda value: value.name)
        ]

    def list_openai_tools(self, selected_names: list[str] | None = None) -> list[dict[str, Any]]:
        selected = {item.strip() for item in selected_names or [] if item and item.strip()}
        return [
            {
                "type": "function",
                "function": {
                    "name": item.name,
                    "description": item.description,
                    "parameters": item.input_schema,
                },
            }
            for item in sorted(self._tools.values(), key=lambda value: value.name)
            if (not selected or item.name in selected) and self.is_allowed(item.name)
        ]

    def monitor_snapshot(self) -> dict[str, Any]:
        loaded = sorted(self._tools.keys())
        if self._allowed_tools is None:
            allowed = loaded
            denied: list[str] = []
        else:
            allowed = [item for item in loaded if item in self._allowed_tools]
            denied = [item for item in loaded if item not in self._allowed_tools]
        return {
            "loaded_tools": loaded,
            "allowed_tools": allowed,
            "denied_tools": denied,
            "plugin_load_errors": list(self._plugin_load_errors),
            "recent_events": list(self._status_events)[-50:],
        }


def build_mcp_query_tool(rag_client: RAGClient) -> Callable[..., Any]:
    async def mcp_query_tool(
        query: str,
        top_k: int = 5,
        collection: str | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        turn_id: int = 0,
    ) -> dict[str, Any]:
        return await rag_client.query(
            question=query,
            top_k=top_k,
            collection=collection,
            trace_id=trace_id,
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
        )

    return mcp_query_tool


def build_mcp_list_collections_tool(rag_client: RAGClient) -> Callable[..., Any]:
    async def mcp_list_collections_tool(
        include_stats: bool = True,
        trace_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        turn_id: int = 0,
    ) -> dict[str, Any]:
        return await rag_client.call_tool(
            tool_name="list_collections",
            arguments={"include_stats": bool(include_stats)},
            trace_id=trace_id,
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
        )

    return mcp_list_collections_tool


def build_mcp_get_document_summary_tool(rag_client: RAGClient) -> Callable[..., Any]:
    async def mcp_get_document_summary_tool(
        doc_id: str,
        collection: str | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        turn_id: int = 0,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {"doc_id": str(doc_id).strip()}
        if collection:
            arguments["collection"] = collection
        return await rag_client.call_tool(
            tool_name="get_document_summary",
            arguments=arguments,
            trace_id=trace_id,
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
        )

    return mcp_get_document_summary_tool
