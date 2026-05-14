from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import LLMProfile, Settings
from app.conversation_agent import ConversationAgent
from app.providers import LLMResult


class ConversationFallbackTests(unittest.TestCase):
    def _build_agent(self) -> ConversationAgent:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        settings = Settings(
            rag_server_url="http://127.0.0.1:7300",
            local_profile=LLMProfile(name="local", base_url="http://127.0.0.1:8000/v1", api_key="x", model="m"),
            online_profile=LLMProfile(name="online", base_url="http://127.0.0.1:8001/v1", api_key="x", model="m"),
            default_profile="local",
            user_id="u1",
            session_id="s1",
            max_context_tokens=30000,
            compression_strategy="summary",
            observability_dir=str(root / "obs"),
            history_dir=str(root / "his"),
        )
        agent = ConversationAgent(settings)
        self.addCleanup(agent.close)
        return agent

    def test_fallback_reply_when_llm_unavailable(self) -> None:
        agent = self._build_agent()
        with patch("app.providers.OpenAICompatibleProvider.generate", side_effect=RuntimeError("connect failed")):
            reply = agent.ask_sync("你好")
        self.assertIn("fallback mode", reply.content)
        self.assertEqual(reply.prompt_tokens, 0)
        self.assertEqual(reply.completion_tokens, 0)

    def test_llm_driven_tool_calling(self) -> None:
        agent = self._build_agent()
        calls: list[dict[str, object]] = []

        def fake_generate(messages, temperature: float = 0.2, tools=None, tool_choice=None):
            calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
            if len(calls) == 1:
                return LLMResult(
                    content="",
                    prompt_tokens=11,
                    completion_tokens=4,
                    raw={},
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "mcp_query_knowledge_hub",
                                "arguments": "{\"query\":\"奥迪销量\"}",
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                )
            return LLMResult(
                content="根据检索结果，当前未命中相关文档。",
                prompt_tokens=7,
                completion_tokens=6,
                raw={},
            )

        async def fake_tool(**kwargs):
            return {"content": f"tool:{kwargs.get('query', '')}", "is_error": False}

        with patch("app.providers.OpenAICompatibleProvider.generate", side_effect=fake_generate):
            with patch.object(agent._tool_registry, "get", return_value=fake_tool):
                reply = agent.ask_sync("请帮我查奥迪销量")

        self.assertTrue(reply.tool_used)
        self.assertIn("未命中相关文档", reply.content)
        self.assertTrue(calls[0]["tools"])
        self.assertEqual(calls[0]["tool_choice"], "auto")
        self.assertEqual(len(calls), 2)

    def test_tool_permission_denied_is_standardized(self) -> None:
        agent = self._build_agent()
        agent._tool_registry.set_allowed_tools(["not_mcp_query_knowledge_hub"])

        calls: list[dict[str, object]] = []

        def fake_generate(messages, temperature: float = 0.2, tools=None, tool_choice=None):
            calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
            if len(calls) == 1:
                return LLMResult(
                    content="",
                    prompt_tokens=8,
                    completion_tokens=3,
                    raw={},
                    tool_calls=[
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "mcp_query_knowledge_hub",
                                "arguments": "{\"query\":\"奥迪销量\"}",
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                )
            return LLMResult(content="执行完成", prompt_tokens=6, completion_tokens=2, raw={})

        with patch("app.providers.OpenAICompatibleProvider.generate", side_effect=fake_generate):
            with patch.object(agent._tool_registry, "get", side_effect=AssertionError("tool handler should not be called")):
                reply = agent.ask_sync("请检索奥迪销量")

        self.assertEqual(reply.tool_payload.get("status"), "error")
        self.assertEqual(reply.tool_payload.get("error", {}).get("code"), "tool_permission_denied")
        self.assertEqual(reply.content, "执行完成")

    def test_dynamic_tool_binding_can_filter_runtime_tools(self) -> None:
        agent = self._build_agent()
        calls: list[dict[str, object]] = []

        def fake_generate(messages, temperature: float = 0.2, tools=None, tool_choice=None):
            calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
            return LLMResult(content="直接回答", prompt_tokens=4, completion_tokens=2, raw={})

        with patch.object(
            agent,
            "_recognize_tool_intent",
            return_value={"intent": "focused", "candidate_tools": ["not_exists"]},
        ):
            with patch("app.providers.OpenAICompatibleProvider.generate", side_effect=fake_generate):
                reply = agent.ask_sync("你好")

        self.assertEqual(reply.content, "直接回答")
        self.assertIn(calls[0]["tools"], (None, []))
        self.assertEqual(calls[0]["tool_choice"], "auto")

    def test_tool_message_is_persisted_with_tool_call_id(self) -> None:
        agent = self._build_agent()

        def fake_generate(messages, temperature: float = 0.2, tools=None, tool_choice=None):
            if not any(item.get("role") == "tool" for item in messages):
                return LLMResult(
                    content="",
                    prompt_tokens=8,
                    completion_tokens=3,
                    raw={},
                    tool_calls=[
                        {
                            "id": "call_3",
                            "type": "function",
                            "function": {
                                "name": "mcp_query_knowledge_hub",
                                "arguments": "{\"query\":\"奥迪销量\"}",
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                )
            return LLMResult(content="执行完成", prompt_tokens=6, completion_tokens=2, raw={})

        async def fake_tool(**kwargs):
            return {"content": f"tool:{kwargs.get('query', '')}", "is_error": False}

        with patch("app.providers.OpenAICompatibleProvider.generate", side_effect=fake_generate):
            with patch.object(agent._tool_registry, "get", return_value=fake_tool):
                reply = agent.ask_sync("请检索奥迪销量")

        self.assertEqual(reply.content, "执行完成")
        tool_messages = [item for item in agent.state.messages if item.get("role") == "tool"]
        self.assertTrue(tool_messages)
        self.assertEqual(tool_messages[-1].get("tool_call_id"), "call_3")

    def test_tool_runtime_status_supports_load_and_unload_feedback(self) -> None:
        agent = self._build_agent()
        unloaded = agent._tool_registry.unregister("mcp_query_knowledge_hub")
        self.assertTrue(unloaded)
        status = agent.tool_runtime_status()
        self.assertIn("loaded_tools", status)
        self.assertIn("allowed_tools", status)
        self.assertIn("recent_events", status)
        events = status["recent_events"]
        self.assertTrue(any(item.get("event_type") == "unload" for item in events))

    def test_default_runtime_tools_include_three_mcp_tools(self) -> None:
        agent = self._build_agent()
        tool_names = set(agent.list_tools())
        self.assertIn("mcp_query_knowledge_hub", tool_names)
        self.assertIn("mcp_list_collections", tool_names)
        self.assertIn("mcp_get_document_summary", tool_names)

    def test_complex_request_prefers_online_provider_in_react_mode(self) -> None:
        agent = self._build_agent()
        selected_providers: list[str] = []

        def fake_generate_with_provider(provider_name, prompt_messages, tools=None, tool_choice=None):
            selected_providers.append(provider_name)
            return LLMResult(content="ok", prompt_tokens=5, completion_tokens=3, raw={})

        with patch.object(agent, "_generate_with_provider", side_effect=fake_generate_with_provider):
            reply = agent.ask_sync("请分析这份复杂报告并结合检索结果给出诊断结论和建议")
        self.assertEqual(reply.content, "ok")
        self.assertTrue(selected_providers)
        self.assertEqual(selected_providers[0], "online")


if __name__ == "__main__":
    unittest.main()
