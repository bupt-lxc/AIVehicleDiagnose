from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import main


class _FakeAgent:
    def __init__(self) -> None:
        self.active_provider_name = "online"
        self.state = SimpleNamespace(to_dict=lambda: {"ok": True})
        self._asked: list[str] = []

    def describe_tools(self) -> list[dict[str, str]]:
        return [{"name": "mcp_query_knowledge_hub", "description": "demo"}]

    def switch_provider(self, provider_name: str, model: str | None = None) -> str:
        self.active_provider_name = provider_name
        return provider_name

    def ask_sync(self, raw: str) -> SimpleNamespace:
        self._asked.append(raw)
        return SimpleNamespace(
            trace_id="trace-test",
            tool_used=False,
            prompt_tokens=3,
            completion_tokens=2,
            content=f"echo:{raw}",
            tool_payload={"content": ""},
        )

    def export_conversation(self) -> dict[str, str]:
        return {"conversation": "ok"}

    def delete_conversation(self) -> None:
        return None


class MainCliSimulationTests(unittest.TestCase):
    def test_chat_loop_simulates_user_switch_and_ask(self) -> None:
        agent = _FakeAgent()
        graph = main._UnavailableGraph("graph disabled")
        stdout = io.StringIO()
        user_inputs = ["/profile", "/switch local", "您好", "/exit"]
        with patch("builtins.input", side_effect=user_inputs):
            with redirect_stdout(stdout):
                main.chat_loop(agent, graph)
        output = stdout.getvalue()
        self.assertIn("Interactive mode started.", output)
        self.assertIn("Active provider: online", output)
        self.assertIn("Switched to provider=local", output)
        self.assertIn("Trace ID: trace-test", output)
        self.assertIn("echo:您好", output)
        self.assertEqual(agent._asked, ["您好"])

    def test_chat_loop_simulates_state_and_tools_commands(self) -> None:
        agent = _FakeAgent()
        graph = main._UnavailableGraph("graph disabled")
        stdout = io.StringIO()
        user_inputs = ["/tools", "/state", "/exit"]
        with patch("builtins.input", side_effect=user_inputs):
            with redirect_stdout(stdout):
                main.chat_loop(agent, graph)
        output = stdout.getvalue()
        self.assertIn("mcp_query_knowledge_hub", output)
        self.assertIn('"ok": true', output.lower())


if __name__ == "__main__":
    unittest.main()
