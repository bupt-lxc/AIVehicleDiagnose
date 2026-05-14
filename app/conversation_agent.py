from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .agent_tools import (
    ToolRegistry,
    build_mcp_get_document_summary_tool,
    build_mcp_list_collections_tool,
    build_mcp_query_tool,
)
from .compression import CompressionRegistry
from .config import Settings
from .observability import AuditContext, AuditLogger
from .providers import OpenAICompatibleProvider, ProviderRegistry
from .rag_client import RAGClient
from .session import ContextManager, ConversationState, ConversationStore


@dataclass(slots=True)
class AgentReply:
    trace_id: str
    content: str
    tool_used: bool
    tool_payload: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int


class ConversationAgent:
    SYSTEM_PROMPT = (
        "You are an AI Agent assistant following a ReAct workflow. "
        "Reason about the task, call the most suitable tools when needed, then provide a grounded final answer."
    )
    SYSTEM_PROMPT_VERSION = "v1.1.0"
    MAX_TOOL_CALL_ROUNDS = 3
    DEFAULT_TOOL_CALL_TIMEOUT_SECONDS = 20.0

    def __init__(self, settings: Settings) -> None:
        dirs = settings.ensure_runtime_dirs()
        self._settings = settings
        self._audit_logger = AuditLogger(dirs["observability"])
        self._store = ConversationStore(dirs["history"])
        self._compression_registry = CompressionRegistry()
        self._compression_registry.load_plugins(settings.compression_plugins)
        self._context_manager = ContextManager(self._compression_registry, self._audit_logger)
        self._rag_client = RAGClient(settings=settings, audit_logger=self._audit_logger)
        self._tool_registry = ToolRegistry()
        self._tool_registry.register(
            "mcp_query_knowledge_hub",
            build_mcp_query_tool(self._rag_client),
            description=(
                "Query enterprise knowledge hub through MCP RAG pipeline. "
                "Use for retrieval questions such as sales metrics, diagnostics evidence, "
                "document lookup, references, and fact-grounded answers."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User question for retrieval."},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                    "collection": {"type": ["string", "null"], "default": None},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )
        self._tool_registry.register(
            "mcp_list_collections",
            build_mcp_list_collections_tool(self._rag_client),
            description=(
                "List all available MCP collections with optional statistics. "
                "Use when you need collection discovery before retrieval."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "include_stats": {"type": "boolean", "default": True},
                },
                "required": [],
                "additionalProperties": False,
            },
        )
        self._tool_registry.register(
            "mcp_get_document_summary",
            build_mcp_get_document_summary_tool(self._rag_client),
            description=(
                "Fetch summary and metadata for a specific document by document id. "
                "Use after retrieval when document-level details are required."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "collection": {"type": ["string", "null"], "default": None},
                },
                "required": ["doc_id"],
                "additionalProperties": False,
            },
        )
        self._tool_registry.load_plugins(settings.tool_plugins)
        allowed_tools = [item.strip() for item in os.getenv("ALLOWED_TOOL_NAMES", "").split(",") if item.strip()]
        self._tool_registry.set_allowed_tools(allowed_tools or None)
        timeout_raw = os.getenv("TOOL_CALL_TIMEOUT_SECONDS", str(self.DEFAULT_TOOL_CALL_TIMEOUT_SECONDS)).strip()
        try:
            timeout_value = float(timeout_raw)
        except Exception:
            timeout_value = self.DEFAULT_TOOL_CALL_TIMEOUT_SECONDS
        self._tool_call_timeout_seconds = timeout_value if timeout_value > 0 else self.DEFAULT_TOOL_CALL_TIMEOUT_SECONDS
        self._provider_registry = ProviderRegistry()
        self._provider_registry.register("local", OpenAICompatibleProvider(settings.local_profile))
        self._provider_registry.register("online", OpenAICompatibleProvider(settings.online_profile))
        self._provider_registry.load_plugins(settings.llm_provider_plugins, settings.local_profile)
        self._state = self._store.load_or_create(
            user_id=settings.user_id,
            session_id=settings.session_id,
            max_context_tokens=settings.max_context_tokens,
            compression_strategy=settings.compression_strategy,
        )
        default_provider = settings.default_profile if settings.default_profile in settings.profiles else "online"
        self._active_provider_name = default_provider

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def active_provider_name(self) -> str:
        return self._active_provider_name

    def switch_provider(self, provider_name: str, model: str | None = None) -> str:
        normalized = provider_name.strip().lower()
        if normalized in self._settings.profiles and model:
            self._settings.profiles[normalized].model = model.strip()
            provider = OpenAICompatibleProvider(self._settings.profiles[normalized])
            self._provider_registry.register(normalized, provider)
        self._provider_registry.get(normalized)
        self._active_provider_name = normalized
        return normalized

    def list_tools(self) -> list[str]:
        return self._tool_registry.list_tools()

    def describe_tools(self) -> list[dict[str, Any]]:
        return self._tool_registry.describe_tools()

    def tool_runtime_status(self) -> dict[str, Any]:
        snapshot = self._tool_registry.monitor_snapshot()
        snapshot["tool_call_timeout_seconds"] = self._tool_call_timeout_seconds
        return snapshot

    def _normalize_tool_calls(self, raw_tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in raw_tool_calls:
            if not isinstance(item, dict):
                continue
            function_payload = item.get("function", {})
            if not isinstance(function_payload, dict):
                continue
            tool_name = str(function_payload.get("name", "")).strip()
            if not tool_name:
                continue
            arguments_raw = function_payload.get("arguments", "{}")
            arguments = arguments_raw if isinstance(arguments_raw, str) else json.dumps(arguments_raw, ensure_ascii=False)
            normalized.append(
                {
                    "id": str(item.get("id", "")).strip() or f"tool_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": arguments},
                }
            )
        return normalized

    def _build_tool_message_content(self, tool_result: dict[str, Any]) -> str:
        content = tool_result.get("content", "")
        if isinstance(content, str) and content.strip():
            return content
        return json.dumps(tool_result, ensure_ascii=False)

    def _recognize_tool_intent(self, user_input: str, prompt_messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "intent": "default",
            "candidate_tools": None,
            "user_input_length": len(user_input),
            "context_messages": len(prompt_messages),
        }

    def _inject_runtime_tools(self, intent_result: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        selected_names_raw = intent_result.get("candidate_tools")
        selected_names: list[str] | None = None
        if isinstance(selected_names_raw, list):
            selected_names = [str(item).strip() for item in selected_names_raw if str(item).strip()]
        tools = self._tool_registry.list_openai_tools(selected_names=selected_names)
        status = self._tool_registry.monitor_snapshot()
        status["injected_tool_names"] = [item.get("function", {}).get("name", "") for item in tools]
        status["intent"] = intent_result.get("intent", "default")
        return tools, status

    def _standardize_tool_result(
        self,
        *,
        tool_name: str,
        request_payload: dict[str, Any],
        raw_result: Any = None,
        is_error: bool = False,
        error_code: str = "",
        error_message: str = "",
        elapsed_ms: float = 0.0,
    ) -> dict[str, Any]:
        if isinstance(raw_result, dict):
            content_text = str(raw_result.get("content", "")).strip()
            citations = raw_result.get("citations", [])
        else:
            content_text = str(raw_result or "").strip()
            citations = []
        if not content_text:
            content_text = "Tool execution finished." if not is_error else (error_message or "Tool execution failed.")
        payload = {
            "tool_name": tool_name,
            "status": "error" if is_error else "ok",
            "is_error": bool(is_error),
            "content": content_text,
            "citations": citations if isinstance(citations, list) else [],
            "raw": raw_result if isinstance(raw_result, dict) else {"value": raw_result},
            "error": {"code": error_code, "message": error_message} if is_error else None,
            "meta": {"elapsed_ms": round(float(elapsed_ms), 2), "request": request_payload},
        }
        return payload

    def _prepare_tool_request(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        trace_id: str,
    ) -> dict[str, Any]:
        request_payload = dict(arguments)
        if tool_name == "mcp_query_knowledge_hub":
            request_payload["query"] = str(request_payload.get("query", "")).strip()
            try:
                request_payload["top_k"] = int(request_payload.get("top_k", 5) or 5)
            except Exception:
                request_payload["top_k"] = 5
            collection_value = request_payload.get("collection")
            request_payload["collection"] = None if collection_value in {"", None} else collection_value
            request_payload["trace_id"] = trace_id
            request_payload["user_id"] = self._state.user_id
            request_payload["session_id"] = self._state.session_id
            request_payload["turn_id"] = self._state.turn_index
        elif tool_name == "mcp_list_collections":
            request_payload["include_stats"] = bool(request_payload.get("include_stats", True))
            request_payload["trace_id"] = trace_id
            request_payload["user_id"] = self._state.user_id
            request_payload["session_id"] = self._state.session_id
            request_payload["turn_id"] = self._state.turn_index
        elif tool_name == "mcp_get_document_summary":
            request_payload["doc_id"] = str(request_payload.get("doc_id", "")).strip()
            collection_value = request_payload.get("collection")
            request_payload["collection"] = None if collection_value in {"", None} else collection_value
            request_payload["trace_id"] = trace_id
            request_payload["user_id"] = self._state.user_id
            request_payload["session_id"] = self._state.session_id
            request_payload["turn_id"] = self._state.turn_index
        return request_payload

    def _select_react_provider(
        self,
        *,
        user_input: str,
        has_tools: bool,
    ) -> str:
        normalized = user_input.strip().lower()
        complex_keywords = (
            "分析",
            "诊断",
            "复杂",
            "compare",
            "分析一下",
            "step",
            "report",
            "tool",
            "检索",
            "总结",
            "方案",
        )
        is_complex = len(normalized) >= 36 or any(item in normalized for item in complex_keywords)
        if has_tools and any(item in normalized for item in ("检索", "文档", "collection", "summary", "citation")):
            is_complex = True
        if is_complex and self._can_use_online_provider():
            return "online"
        return self._active_provider_name

    async def _execute_tool_call(
        self,
        *,
        tool_call: dict[str, Any],
        trace_id: str,
        round_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        started_at = self._store._utcnow()
        track_id = f"trk_{uuid.uuid4().hex}"
        tool_call_id = str(tool_call.get("id", "")).strip() or f"tool_{uuid.uuid4().hex}"
        function_payload = tool_call.get("function", {})
        tool_name = str(function_payload.get("name", "")).strip()
        raw_arguments = function_payload.get("arguments", "{}")
        parsed_arguments: dict[str, Any] = {}
        arguments_error = ""
        if isinstance(raw_arguments, str):
            try:
                loaded = json.loads(raw_arguments or "{}")
                parsed_arguments = loaded if isinstance(loaded, dict) else {}
            except Exception:
                parsed_arguments = {}
                arguments_error = "invalid_tool_arguments_json"
        elif isinstance(raw_arguments, dict):
            parsed_arguments = dict(raw_arguments)
        elif raw_arguments is not None:
            arguments_error = "invalid_tool_arguments_type"
        request_payload = self._prepare_tool_request(tool_name=tool_name, arguments=parsed_arguments, trace_id=trace_id)
        tool_result: dict[str, Any]
        error_text = ""
        elapsed_ms = 0.0
        if not self._tool_registry.is_allowed(tool_name):
            error_text = f"Tool call denied by policy: {tool_name}"
            tool_result = self._standardize_tool_result(
                tool_name=tool_name or "unknown_tool",
                request_payload=request_payload,
                raw_result={},
                is_error=True,
                error_code="tool_permission_denied",
                error_message=error_text,
                elapsed_ms=elapsed_ms,
            )
            finished_at = self._store._utcnow()
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": self._build_tool_message_content(tool_result),
            }
            tool_record = {
                "tool_call_id": tool_call_id,
                "track_id": track_id,
                "round": round_index,
                "tool_name": tool_name or "unknown_tool",
                "request": request_payload,
                "result": tool_result,
                "status": "error",
                "started_at": started_at,
                "finished_at": finished_at,
                "error": error_text,
            }
            return tool_message, tool_record, tool_result
        if arguments_error:
            error_text = f"Tool arguments parse failed: {arguments_error}"
            tool_result = self._standardize_tool_result(
                tool_name=tool_name or "unknown_tool",
                request_payload=request_payload,
                raw_result={},
                is_error=True,
                error_code=arguments_error,
                error_message=error_text,
                elapsed_ms=elapsed_ms,
            )
            finished_at = self._store._utcnow()
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": self._build_tool_message_content(tool_result),
            }
            tool_record = {
                "tool_call_id": tool_call_id,
                "track_id": track_id,
                "round": round_index,
                "tool_name": tool_name or "unknown_tool",
                "request": request_payload,
                "result": tool_result,
                "status": "error",
                "started_at": started_at,
                "finished_at": finished_at,
                "error": error_text,
            }
            return tool_message, tool_record, tool_result
        try:
            call_started = time.perf_counter()
            tool_handler = self._tool_registry.get(tool_name)
            
            import inspect
            if inspect.iscoroutinefunction(tool_handler):
                raw_result = await asyncio.wait_for(
                    tool_handler(**request_payload),
                    timeout=self._tool_call_timeout_seconds,
                )
            else:
                loop = asyncio.get_running_loop()
                raw_result = await loop.run_in_executor(
                    None,
                    lambda: tool_handler(**request_payload)
                )
            elapsed_ms = (time.perf_counter() - call_started) * 1000
            tool_result = self._standardize_tool_result(
                tool_name=tool_name or "unknown_tool",
                request_payload=request_payload,
                raw_result=raw_result,
                is_error=False,
                elapsed_ms=elapsed_ms,
            )
        except asyncio.TimeoutError:
            elapsed_ms = self._tool_call_timeout_seconds * 1000
            error_text = f"Tool call timeout after {self._tool_call_timeout_seconds:.2f}s"
            tool_result = self._standardize_tool_result(
                tool_name=tool_name or "unknown_tool",
                request_payload=request_payload,
                raw_result={},
                is_error=True,
                error_code="tool_timeout",
                error_message=error_text,
                elapsed_ms=elapsed_ms,
            )
        except Exception as exc:
            error_text = str(exc)
            tool_result = self._standardize_tool_result(
                tool_name=tool_name or "unknown_tool",
                request_payload=request_payload,
                raw_result={},
                is_error=True,
                error_code="tool_execution_failed",
                error_message=error_text,
                elapsed_ms=elapsed_ms,
            )
        finished_at = self._store._utcnow()
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": self._build_tool_message_content(tool_result),
        }
        tool_record = {
            "tool_call_id": tool_call_id,
            "track_id": track_id,
            "round": round_index,
            "tool_name": tool_name or "unknown_tool",
            "request": request_payload,
            "result": tool_result,
            "status": "ok" if not tool_result.get("is_error", False) else "error",
            "started_at": started_at,
            "finished_at": finished_at,
            "error": error_text,
        }
        return tool_message, tool_record, tool_result

    def _emit_turn_event(self, trace_id: str, event_type: str, payload: dict[str, Any]) -> None:
        self._audit_logger.emit(
            event_type,
            AuditContext(
                user_id=self._state.user_id,
                session_id=self._state.session_id,
                trace_id=trace_id,
                turn_id=self._state.turn_index,
                metadata={"provider": self._active_provider_name},
            ),
            payload,
        )

    def _build_fallback_response(self, user_input: str, tool_payload: dict[str, Any]) -> str:
        tool_content = str(tool_payload.get("content", "")).strip()
        if tool_content:
            return (
                "LLM backend is currently unavailable, switched to fallback mode.\n"
                f"User question: {user_input}\n"
                "Retrieved tool result:\n"
                f"{tool_content[:1500]}"
            )
        return (
            "LLM backend is currently unavailable, switched to fallback mode.\n"
            "No tool context was available for this turn.\n"
            "You can use /switch online to try another provider."
        )

    def _can_use_online_provider(self) -> bool:
        return bool(self._settings.online_profile.api_key.strip())

    def _generate_with_provider(
        self,
        provider_name: str,
        prompt_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ):
        provider = self._provider_registry.get(provider_name)
        return provider.generate(prompt_messages, tools=tools, tool_choice=tool_choice)

    def _generate_with_failover(
        self,
        *,
        trace_id: str,
        prompt_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        preferred_provider: str | None = None,
    ) -> tuple[Any, str]:
        primary_provider = preferred_provider or self._active_provider_name
        used_provider = primary_provider
        try:
            llm_result = self._generate_with_provider(
                primary_provider,
                prompt_messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            return llm_result, used_provider
        except Exception as exc:
            if primary_provider == "local" and self._can_use_online_provider():
                llm_result = self._generate_with_provider(
                    "online",
                    prompt_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                self._emit_turn_event(
                    trace_id,
                    "llm_provider_failover",
                    {
                        "from_provider": primary_provider,
                        "to_provider": "online",
                        "reason": str(exc),
                    },
                )
                self._active_provider_name = "online"
                return llm_result, "online"
            if primary_provider == "online":
                llm_result = self._generate_with_provider(
                    "local",
                    prompt_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                self._emit_turn_event(
                    trace_id,
                    "llm_provider_failover",
                    {
                        "from_provider": primary_provider,
                        "to_provider": "local",
                        "reason": str(exc),
                    },
                )
                self._active_provider_name = "local"
                return llm_result, "local"
            raise

    async def ask(self, user_input: str) -> AgentReply:
        self._state.turn_index += 1
        trace_id = str(uuid.uuid4())
        processing_steps: list[dict[str, Any]] = []
        processing_steps.append(
            {
                "step_type": "context_snapshot_before",
                "input": {"turn_index": self._state.turn_index},
                "output": {
                    "summary_memory": self._state.summary_memory,
                    "messages_size": len(self._state.messages),
                },
                "metadata": {"trace_id": trace_id},
            }
        )
        self._context_manager.maybe_compress(self._state, trace_id=trace_id)
        normalized_user_input = user_input
        self._state.messages.append({"role": "user", "content": normalized_user_input})
        prompt_messages = self._context_manager.build_prompt_messages(self._state)
        if prompt_messages and prompt_messages[0].get("role") == "system":
            prompt_messages[0] = {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            }
        intent_result = self._recognize_tool_intent(normalized_user_input, prompt_messages)
        tools, tool_injection_status = self._inject_runtime_tools(intent_result)
        react_provider = self._select_react_provider(
            user_input=normalized_user_input,
            has_tools=bool(tools),
        )
        working_messages: list[dict[str, Any]] = [dict(item) for item in prompt_messages]
        tool_records: list[dict[str, Any]] = []
        tool_payload: dict[str, Any] = {}
        tool_used = False
        prompt_tokens = 0
        completion_tokens = 0
        used_provider = self._active_provider_name
        llm_raw: dict[str, Any] = {}
        assistant_content = ""
        processing_steps.append(
            {
                "step_type": "prompt_assembled",
                "input": {"tool_calling_mode": "react_node"},
                "output": {
                    "prompt_messages": working_messages,
                    "tools": tools,
                    "tool_injection_status": tool_injection_status,
                    "react_provider": react_provider,
                },
                "metadata": {"prompt_size": len(prompt_messages)},
            }
        )
        try:
            for round_index in range(1, self.MAX_TOOL_CALL_ROUNDS + 1):
                llm_result, used_provider = self._generate_with_failover(
                    trace_id=trace_id,
                    prompt_messages=working_messages,
                    tools=tools or None,
                    tool_choice="auto",
                    preferred_provider=react_provider,
                )
                prompt_tokens += int(getattr(llm_result, "prompt_tokens", 0) or 0)
                completion_tokens += int(getattr(llm_result, "completion_tokens", 0) or 0)
                llm_raw_payload = getattr(llm_result, "raw", {})
                llm_raw = llm_raw_payload if isinstance(llm_raw_payload, dict) else {}
                raw_tool_calls = getattr(llm_result, "tool_calls", [])
                normalized_tool_calls = self._normalize_tool_calls(raw_tool_calls if isinstance(raw_tool_calls, list) else [])
                llm_content = str(getattr(llm_result, "content", "") or "")
                llm_finish_reason = str(getattr(llm_result, "finish_reason", "stop") or "stop")
                processing_steps.append(
                    {
                        "step_type": "llm_round",
                        "input": {
                            "round": round_index,
                            "provider": used_provider,
                            "tool_choice": "auto",
                        },
                        "output": {
                            "finish_reason": llm_finish_reason,
                            "tool_calls": normalized_tool_calls,
                            "assistant_preview": llm_content[:200],
                        },
                        "metadata": {"trace_id": trace_id},
                    }
                )
                if not normalized_tool_calls:
                    assistant_content = llm_content.strip() or "No response generated."
                    break
                tool_used = True
                working_messages.append(
                    {
                        "role": "assistant",
                        "content": llm_content,
                        "tool_calls": normalized_tool_calls,
                    }
                )
                for tool_call in normalized_tool_calls:
                    tool_message, tool_record, tool_response = await self._execute_tool_call(
                        tool_call=tool_call,
                        trace_id=trace_id,
                        round_index=round_index,
                    )
                    tool_records.append(tool_record)
                    tool_payload = tool_response
                    self._state.messages.append(
                        {
                            "role": "tool",
                            "content": tool_message["content"],
                            "tool_call_id": tool_record["tool_call_id"],
                        }
                    )
                    working_messages.append(tool_message)
                    self._emit_turn_event(
                        trace_id,
                        "tool_attached",
                        {
                            "tool_name": tool_record["tool_name"],
                            "tool_is_error": tool_response.get("is_error", False),
                            "tool_call_id": tool_record["tool_call_id"],
                        },
                    )
                if round_index >= self.MAX_TOOL_CALL_ROUNDS:
                    assistant_content = llm_content.strip() or "Tool call round limit reached."
            if not assistant_content:
                assistant_content = "No response generated."
        except Exception as exc:
            assistant_content = self._build_fallback_response(normalized_user_input, tool_payload)
            self._emit_turn_event(
                trace_id,
                "llm_call_error",
                {
                    "provider": self._active_provider_name,
                    "error": str(exc),
                    "fallback_enabled": True,
                },
            )
        processing_steps.append(
            {
                "step_type": "llm_generation",
                "input": {"provider": used_provider, "tool_used": tool_used},
                "output": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "assistant_preview": assistant_content[:200],
                },
                "metadata": {"trace_id": trace_id},
            }
        )
        self._state.messages.append({"role": "assistant", "content": assistant_content})
        self._context_manager.maybe_compress(self._state, trace_id=trace_id)
        assistant_metadata = {
            "provider": used_provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "trace_id": trace_id,
            "raw": llm_raw,
        }
        self._store.record_turn(
            state=self._state,
            trace_id=trace_id,
            system_prompt=self.SYSTEM_PROMPT,
            system_prompt_version=self.SYSTEM_PROMPT_VERSION,
            user_text=normalized_user_input,
            assistant_text=assistant_content,
            prompt_messages=working_messages,
            tool_calls=tool_records,
            processing_steps=processing_steps,
            assistant_metadata=assistant_metadata,
        )
        self._store.save_state(self._state)
        self._audit_logger.update_tokens(prompt_tokens, completion_tokens)
        self._emit_turn_event(
            trace_id,
            "agent_reply",
            {
                "tool_used": tool_used,
                "provider": used_provider,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "response_preview": assistant_content[:800],
            },
        )
        return AgentReply(
            trace_id=trace_id,
            content=assistant_content,
            tool_used=tool_used,
            tool_payload=tool_payload,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def export_state_json(self) -> str:
        payload = {
            "state": self._state.to_dict(),
            "tools": self.list_tools(),
            "tool_runtime_status": self.tool_runtime_status(),
            "provider": self._active_provider_name,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def export_conversation(self) -> dict[str, Any]:
        return self._store.export_conversation(self._state.conversation_id)

    def delete_conversation(self) -> None:
        self._store.delete_conversation(self._state.conversation_id)

    def close(self) -> None:
        self._store.close()

    def ask_sync(self, user_input: str) -> AgentReply:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
            
        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot call ask_sync from a running event loop. "
                "Use 'await agent.ask(...)' instead."
            )
        return asyncio.run(self.ask(user_input))
