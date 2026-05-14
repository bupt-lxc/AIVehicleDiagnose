from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.config import LLMProfile, Settings
from app.observability import AuditLogger
from app.rag_client import RAGClient


class FakeSSEClient:
    def __init__(self, url: str) -> None:
        self._url = url

    async def __aenter__(self):
        return object(), object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeClientSession:
    attempts = 0

    def __init__(self, read_stream, write_stream) -> None:
        self._read_stream = read_stream
        self._write_stream = write_stream

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def initialize(self):
        return None

    async def call_tool(self, tool_name, arguments):
        FakeClientSession.attempts += 1
        if FakeClientSession.attempts == 1:
            raise RuntimeError("temporary failure")
        text_item = SimpleNamespace(type="text", text=f"ok:{tool_name}:{arguments.get('trace_id', '')}")
        return SimpleNamespace(content=[text_item], isError=False)


class MCPClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_and_trace_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                rag_server_url="http://127.0.0.1:7300",
                local_profile=LLMProfile(name="local", base_url="", api_key="", model=""),
                online_profile=LLMProfile(name="online", base_url="", api_key="", model=""),
                default_profile="local",
                user_id="u1",
                session_id="s1",
                max_context_tokens=30000,
                compression_strategy="summary",
                mcp_retry_count=1,
                mcp_retry_backoff_ms=1,
                observability_dir=str(Path(temp_dir) / "obs"),
                history_dir=str(Path(temp_dir) / "his"),
            )
            audit_logger = AuditLogger(Path(temp_dir) / "obs")
            client = RAGClient(settings, audit_logger=audit_logger)
            FakeClientSession.attempts = 0
            with patch("app.rag_client.sse_client", side_effect=lambda url: FakeSSEClient(url)):
                with patch("app.rag_client.ClientSession", FakeClientSession):
                    with patch("app.rag_client.logger.warning"):
                        with patch("app.rag_client.logger.exception"):
                            result = await client.query("hello", trace_id="trace-abc", turn_id=1)
            self.assertFalse(result["is_error"])
            self.assertEqual(result["trace_id"], "trace-abc")
            self.assertEqual(result["attempt"], 2)


if __name__ == "__main__":
    unittest.main()
