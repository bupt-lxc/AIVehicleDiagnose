from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

try:
    import httpx
except Exception:
    httpx = None

from .config import LLMProfile


@dataclass(slots=True)
class LLMResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    raw: dict[str, Any]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"


class LLMProvider(Protocol):
    name: str

    def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResult:
        ...


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    total_chars = sum(len(str(item.get("content", ""))) for item in messages)
    return max(1, total_chars // 4) if total_chars else 0


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(self, profile: LLMProfile, timeout_seconds: int = 60) -> None:
        self._profile = profile
        self._timeout_seconds = timeout_seconds
        if httpx is not None:
            self._client = httpx.Client(timeout=self._timeout_seconds)
        else:
            self._client = None

    def __del__(self) -> None:
        if hasattr(self, "_client") and self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass

    def _candidate_models(self) -> list[str]:
        env_models = [item.strip() for item in os.getenv("ONLINE_LLM_FALLBACK_MODELS", "").split(",") if item.strip()]
        defaults = ["qwen-turbo", "qwen-max", "qwen-plus"]
        ordered = [m for m in [self._profile.model] if m]  # Filter empty strings
        for model in env_models + defaults:
            if model not in ordered:
                ordered.append(model)
        
        if not ordered:
             # Fallback if everything is empty, though unlikely given defaults
             return defaults
             
        return ordered

    def _should_try_next_model(self, exc: Any) -> bool:
        status = exc.response.status_code
        if status not in {400, 401, 403, 404, 429}:
            return False
        error_text = exc.response.text.lower()
        markers = ("model", "not support", "exhausted", "quota", "forbidden", "invalid", "arrearage", "access denied", "account")
        return any(marker in error_text for marker in markers)

    def _post_completion(
        self,
        client: Any,
        url: str,
        headers: dict[str, str],
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        response = client.post(url, headers=headers, content=json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        response.raise_for_status()
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {}

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in messages:
            role = str(item.get("role", "user")).strip().lower()
            if role == "tool":
                content = str(item.get("content", "")).strip()
                if not content:
                    continue
                tool_call_id = str(item.get("tool_call_id", "")).strip()
                if not tool_call_id:
                    continue
                normalized_item: dict[str, Any] = {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": tool_call_id,
                }
                normalized.append(normalized_item)
                continue
            if role == "assistant":
                content_value = item.get("content")
                content = content_value if isinstance(content_value, str) else ""
                tool_calls = item.get("tool_calls")
                normalized_item = {"role": "assistant", "content": content}
                if isinstance(tool_calls, list) and tool_calls:
                    normalized_item["tool_calls"] = tool_calls
                if not content and not normalized_item.get("tool_calls"):
                    continue
                normalized.append(normalized_item)
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if role not in {"system", "user"}:
                role = "user"
            normalized.append({"role": role, "content": content})
        return normalized

    def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResult:
        import time
        if httpx is None:
            raise RuntimeError("httpx is required for OpenAICompatibleProvider.generate")
        url = f"{self._profile.base_url.rstrip('/')}/chat/completions"
        normalized_messages = self._normalize_messages(messages)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._profile.api_key}",
        }
        data: dict[str, Any] = {}
        final_exc: Exception | None = None
        
        candidates = self._candidate_models()
        start_time = time.time()
        max_total_time = self._timeout_seconds * 2  # 总超时限制
        
        for index, model in enumerate(candidates):
            if time.time() - start_time > max_total_time:
                final_exc = TimeoutError("Total LLM call time exceeded limit")
                break
                
            try:
                data = self._post_completion(
                    self._client,
                    url,
                    headers,
                    model,
                    normalized_messages,
                    temperature,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                if not data or not data.get("choices"):
                    raise ValueError(f"Model {model} returned empty response")
                self._profile.model = model
                final_exc = None
                break
            except Exception as exc:
                if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
                    final_exc = RuntimeError(f"HTTP Error {exc.response.status_code}")
                    if index >= len(candidates) - 1 or not self._should_try_next_model(exc):
                        break
                    continue
                final_exc = RuntimeError(f"API Error: {type(exc).__name__}")
                if index >= len(candidates) - 1:
                    break
        
        if final_exc is not None and not data:
            raise final_exc
            
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("No choices in LLM response")
            
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError(f"Invalid choice format: {type(choice)}")
            
        message = choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("No message in LLM response")
            
        content = ""
        tool_calls: list[dict[str, Any]] = []
        finish_reason = str(choice.get("finish_reason", "stop") or "stop")
        
        content_value = message.get("content")
        content = content_value if isinstance(content_value, str) else ""
        parsed_tool_calls = message.get("tool_calls", [])
        if isinstance(parsed_tool_calls, list):
            tool_calls = [item for item in parsed_tool_calls if isinstance(item, dict)]
            
        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", estimate_tokens(normalized_messages)) or 0)
        completion_tokens = int(usage.get("completion_tokens", max(1, len(content) // 4)) or 0)
        return LLMResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw=data if isinstance(data, dict) else {},
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )


def load_class(path: str) -> Any:
    module_name, class_name = path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    def register(self, name: str, provider: LLMProvider) -> None:
        self._providers[name] = provider

    def load_plugins(self, plugins: list[str], profile: LLMProfile) -> None:
        for item in plugins:
            cls = load_class(item)
            provider = cls(profile)
            provider_name = getattr(provider, "name", cls.__name__)
            self.register(provider_name, provider)

    def get(self, name: str) -> LLMProvider:
        if name not in self._providers:
            raise KeyError(f"LLM provider not found: {name}")
        return self._providers[name]
