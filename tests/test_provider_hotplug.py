from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

try:
    import httpx
except Exception:
    httpx = None
from app.config import LLMProfile
from app.providers import LLMResult, OpenAICompatibleProvider, ProviderRegistry


class DummyProvider:
    name = "dummy-provider"

    def __init__(self, profile: LLMProfile) -> None:
        self._profile = profile

    def generate(self, messages, temperature: float = 0.2) -> LLMResult:
        content = f"{self._profile.name}:{len(messages)}"
        return LLMResult(content=content, prompt_tokens=1, completion_tokens=1, raw={})


@unittest.skipIf(httpx is None, "httpx is not available")
class ProviderHotplugTests(unittest.TestCase):
    def test_load_plugin_provider(self) -> None:
        registry = ProviderRegistry()
        profile = LLMProfile(name="local", base_url="http://127.0.0.1:8000/v1", api_key="x", model="demo")
        registry.load_plugins(["tests.test_provider_hotplug:DummyProvider"], profile)
        provider = registry.get("dummy-provider")
        result = provider.generate([{"role": "user", "content": "hello"}])
        self.assertEqual(result.content, "local:1")

    def test_online_model_fallback(self) -> None:
        profile = LLMProfile(
            name="online",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="x",
            model="qwen-plus",
        )
        provider = OpenAICompatibleProvider(profile)
        forbidden_response = Mock()
        forbidden_response.status_code = 403
        forbidden_response.text = "free tier exhausted"
        forbidden_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "forbidden",
            request=Mock(),
            response=forbidden_response,
        )
        success_response = Mock()
        success_response.raise_for_status.return_value = None
        success_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }
        with patch("app.providers.httpx.Client") as mock_client_cls:
            mock_client = Mock()
            mock_client.post.side_effect = [forbidden_response, success_response]
            mock_client_cls.return_value.__enter__.return_value = mock_client
            result = provider.generate([{"role": "user", "content": "你好"}])
        self.assertEqual(result.content, "ok")
        self.assertEqual(profile.model, "qwen-turbo")

    def test_normalize_messages_drops_tool_without_tool_call_id(self) -> None:
        profile = LLMProfile(
            name="online",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="x",
            model="qwen-plus",
        )
        provider = OpenAICompatibleProvider(profile)
        normalized = provider._normalize_messages(
            [
                {"role": "system", "content": "s"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "content": "{\"ok\":true}"},
                {"role": "tool", "content": "{\"ok\":true}", "tool_call_id": "call_1"},
                {"role": "user", "content": "u"},
            ]
        )
        tool_messages = [item for item in normalized if item.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].get("tool_call_id"), "call_1")


if __name__ == "__main__":
    unittest.main()
