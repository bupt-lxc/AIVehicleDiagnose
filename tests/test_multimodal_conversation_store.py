from __future__ import annotations

import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.config import LLMProfile, Settings
from app.conversation_agent import ConversationAgent
from app.session import ConversationStore


class _FakeVectorCollection:
    def __init__(self) -> None:
        self._rows: dict[str, dict[str, object]] = {}
        self._lock = threading.RLock()

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict[str, object]]) -> None:
        with self._lock:
            for idx, item_id in enumerate(ids):
                self._rows[item_id] = {
                    "id": item_id,
                    "document": documents[idx] if idx < len(documents) else "",
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                }

    def get(self, where: dict[str, object] | None = None) -> dict[str, list[object]]:
        with self._lock:
            rows = list(self._rows.values())
        if where:
            rows = [item for item in rows if all(item["metadata"].get(k) == v for k, v in where.items())]
        return {
            "ids": [item["id"] for item in rows],
            "documents": [item["document"] for item in rows],
            "metadatas": [item["metadata"] for item in rows],
        }

    def delete(self, ids: list[str] | None = None, where: dict[str, object] | None = None) -> None:
        with self._lock:
            if ids:
                for item_id in ids:
                    self._rows.pop(item_id, None)
                return
            if where:
                remove_ids = [item["id"] for item in self._rows.values() if all(item["metadata"].get(k) == v for k, v in where.items())]
                for item_id in remove_ids:
                    self._rows.pop(item_id, None)

    def query(self, query_texts: list[str], n_results: int = 5, where: dict[str, object] | None = None) -> dict[str, list[list[object]]]:
        query = query_texts[0] if query_texts else ""
        with self._lock:
            rows = list(self._rows.values())
        if where:
            rows = [item for item in rows if all(item["metadata"].get(k) == v for k, v in where.items())]
        scored = sorted(rows, key=lambda item: 1 if query in str(item["document"]) else 0, reverse=True)[: max(1, int(n_results))]
        return {
            "ids": [[item["id"] for item in scored]],
            "documents": [[item["document"] for item in scored]],
            "metadatas": [[item["metadata"] for item in scored]],
            "distances": [[0.0 for _ in scored]],
        }


class MultiModalConversationStoreTests(unittest.TestCase):
    def _create_store(self) -> tuple[tempfile.TemporaryDirectory[str], ConversationStore]:
        temp_dir = tempfile.TemporaryDirectory()
        store = ConversationStore(Path(temp_dir.name), vector_collection=_FakeVectorCollection())
        self.addCleanup(temp_dir.cleanup)
        self.addCleanup(store.close)
        return temp_dir, store

    def test_record_turn_export_and_integrity(self) -> None:
        _, store = self._create_store()
        state = store.load_or_create("u1", "s1", 30000, "summary")
        trace_id = str(uuid.uuid4())
        result = store.record_turn(
            state=state,
            trace_id=trace_id,
            system_prompt="system prompt",
            system_prompt_version="v1",
            user_text="用户问题：奥迪销量",
            assistant_text="助手回答：请查看检索结果",
            prompt_messages=[{"role": "system", "content": "system prompt"}, {"role": "user", "content": "用户问题：奥迪销量"}],
            tool_calls=[
                {
                    "tool_name": "mcp_query_knowledge_hub",
                    "round": 1,
                    "request": {"query": "奥迪销量", "top_k": 5},
                    "result": {"content": "tool content", "is_error": False},
                }
            ],
            processing_steps=[
                {
                    "step_type": "prompt_assembled",
                    "input": {"question": "奥迪销量"},
                    "output": {"messages": 2},
                    "metadata": {"stage": "build_prompt"},
                },
                {
                    "step_type": "llm_round",
                    "input": {"round": 1, "provider": "local"},
                    "output": {"finish_reason": "tool_calls", "assistant_preview": "调用工具中"},
                    "metadata": {"stage": "react"},
                }
            ],
            assistant_metadata={"provider": "local", "prompt_tokens": 10, "completion_tokens": 20},
        )
        self.assertEqual(result["conversation_id"], state.conversation_id)
        self.assertEqual(len(result["tool_call_ids"]), 1)
        exported = store.export_conversation(state.conversation_id)
        self.assertEqual(exported["conversation"]["conversation_id"], state.conversation_id)
        self.assertEqual(len(exported["messages"]), 4)
        self.assertEqual(len(exported["tool_calls"]), 1)
        self.assertGreaterEqual(len(exported["vectors"]["ids"]), 4)
        self.assertIn("# Conversation", exported["markdown"])
        self.assertIn("### Round Timeline", exported["markdown"])
        self.assertIn("### Round 1", exported["markdown"])
        self.assertIn("tool_call_result", exported["markdown"])
        queried = store.query_messages(conversation_id=state.conversation_id, role="tool", tool_name="mcp_query_knowledge_hub")
        self.assertEqual(len(queried), 1)
        sem = store.semantic_search("tool content", conversation_id=state.conversation_id, top_k=3)
        self.assertGreaterEqual(len(sem), 1)
        integrity = store.check_integrity(state.conversation_id)
        self.assertTrue(integrity["valid"])

    def test_isolation_and_delete(self) -> None:
        _, store = self._create_store()
        s1 = store.load_or_create("uA", "sess-A", 30000, "summary")
        s2 = store.load_or_create("uB", "sess-B", 30000, "summary")
        store.record_turn(
            state=s1,
            trace_id=str(uuid.uuid4()),
            system_prompt="sp",
            system_prompt_version="v1",
            user_text="alpha question",
            assistant_text="alpha answer",
            prompt_messages=[{"role": "system", "content": "sp"}, {"role": "user", "content": "alpha question"}],
            tool_calls=[],
            processing_steps=[],
            assistant_metadata={"provider": "local"},
        )
        store.record_turn(
            state=s2,
            trace_id=str(uuid.uuid4()),
            system_prompt="sp",
            system_prompt_version="v1",
            user_text="beta question",
            assistant_text="beta answer",
            prompt_messages=[{"role": "system", "content": "sp"}, {"role": "user", "content": "beta question"}],
            tool_calls=[],
            processing_steps=[],
            assistant_metadata={"provider": "local"},
        )
        a_messages = store.query_messages(conversation_id=s1.conversation_id, keyword="beta")
        b_messages = store.query_messages(conversation_id=s2.conversation_id, keyword="beta")
        self.assertEqual(len(a_messages), 0)
        self.assertGreaterEqual(len(b_messages), 1)
        store.delete_conversation(s1.conversation_id)
        with self.assertRaises(ValueError):
            store.export_conversation(s1.conversation_id)
        remain = store.export_conversation(s2.conversation_id)
        self.assertEqual(remain["conversation"]["conversation_id"], s2.conversation_id)

    def test_concurrent_writes(self) -> None:
        _, store = self._create_store()
        state = store.load_or_create("uc", "sc", 30000, "summary")

        def worker(idx: int) -> None:
            store.record_turn(
                state=state,
                trace_id=f"trace-{idx}",
                system_prompt="sp",
                system_prompt_version="v1",
                user_text=f"user-{idx}",
                assistant_text=f"assistant-{idx}",
                prompt_messages=[{"role": "system", "content": "sp"}, {"role": "user", "content": f"user-{idx}"}],
                tool_calls=[],
                processing_steps=[{"step_type": "worker", "input": {"idx": idx}, "output": {"ok": True}, "metadata": {}}],
                assistant_metadata={"provider": "local"},
            )

        threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(8)]
        for item in threads:
            item.start()
        for item in threads:
            item.join()
        exported = store.export_conversation(state.conversation_id)
        self.assertEqual(len(exported["final_outputs"]), 8)
        self.assertEqual(len(exported["messages"]), 24)

    def test_recover_conversation(self) -> None:
        temp_dir, store = self._create_store()
        state = store.load_or_create("ur", "sr", 30000, "summary")
        store.record_turn(
            state=state,
            trace_id="trace-recover",
            system_prompt="sp",
            system_prompt_version="v1",
            user_text="recover question",
            assistant_text="recover answer",
            prompt_messages=[{"role": "system", "content": "sp"}, {"role": "user", "content": "recover question"}],
            tool_calls=[],
            processing_steps=[],
            assistant_metadata={"provider": "local"},
        )
        markdown_path = Path(temp_dir.name) / "multimodal" / "markdown" / f"{state.conversation_id}.md"
        markdown_path.write_text("broken", encoding="utf-8")
        vectors = store.export_conversation(state.conversation_id)["vectors"]["ids"]
        store._vector_collection.delete(ids=vectors)
        recovered = store.recover_conversation(state.conversation_id)
        self.assertTrue(recovered["valid"])
        self.assertIn("Message", markdown_path.read_text(encoding="utf-8"))

    def test_rebuild_markdown_uses_round_layout(self) -> None:
        temp_dir, store = self._create_store()
        state = store.load_or_create("um", "sm", 30000, "summary")
        store.record_turn(
            state=state,
            trace_id="trace-rebuild",
            system_prompt="sp",
            system_prompt_version="v2",
            user_text="请检索文档",
            assistant_text="已完成",
            prompt_messages=[{"role": "system", "content": "sp"}, {"role": "user", "content": "请检索文档"}],
            tool_calls=[
                {
                    "tool_call_id": "call_rebuild_1",
                    "track_id": "trk_rebuild_1",
                    "tool_name": "mcp_query_knowledge_hub",
                    "request": {"query": "文档"},
                    "result": {"content": "ok", "is_error": False},
                }
            ],
            processing_steps=[
                {
                    "step_type": "prompt_assembled",
                    "input": {"q": "文档"},
                    "output": {"prompt_messages": [{"role": "system", "content": "sp"}, {"role": "user", "content": "请检索文档"}]},
                    "metadata": {},
                },
                {
                    "step_type": "llm_round",
                    "input": {"round": 1, "provider": "online"},
                    "output": {
                        "finish_reason": "tool_calls",
                        "assistant_preview": "准备调用工具",
                        "tool_calls": [{"id": "call_rebuild_1"}],
                    },
                    "metadata": {},
                },
            ],
            assistant_metadata={"provider": "online"},
        )
        rebuilt = store.rebuild_markdown(state.conversation_id)
        self.assertEqual(rebuilt["conversation_id"], state.conversation_id)
        markdown_path = Path(temp_dir.name) / "multimodal" / "markdown" / f"{state.conversation_id}.md"
        content = markdown_path.read_text(encoding="utf-8")
        self.assertIn("### Round Timeline", content)
        self.assertIn("### Round 1", content)
        self.assertIn("tool_call_result", content)

    def test_inmemory_vector_rehydration_across_store_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = ConversationStore(root)
            state = store.load_or_create("uv", "sv", 30000, "summary")
            store.record_turn(
                state=state,
                trace_id="trace-rehydrate",
                system_prompt="sp",
                system_prompt_version="v1",
                user_text="vector question",
                assistant_text="vector answer",
                prompt_messages=[{"role": "system", "content": "sp"}, {"role": "user", "content": "vector question"}],
                tool_calls=[],
                processing_steps=[],
                assistant_metadata={"provider": "local"},
            )
            before = store.export_conversation(state.conversation_id)["vectors"]["ids"]
            store.close()
            reopened = ConversationStore(root)
            try:
                after = reopened.export_conversation(state.conversation_id)["vectors"]["ids"]
                self.assertGreaterEqual(len(before), 1)
                self.assertEqual(set(before), set(after))
            finally:
                reopened.close()


class SmokeConversationAgentTests(unittest.TestCase):
    def test_smoke_chat_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings(
                rag_server_url="http://127.0.0.1:7300",
                local_profile=LLMProfile(name="local", base_url="http://127.0.0.1:8000/v1", api_key="x", model="m"),
                online_profile=LLMProfile(name="online", base_url="http://127.0.0.1:8001/v1", api_key="", model="m"),
                default_profile="local",
                user_id="smoke-user",
                session_id="smoke-session",
                max_context_tokens=30000,
                compression_strategy="summary",
                observability_dir=str(root / "obs"),
                history_dir=str(root / "his"),
            )
            agent = ConversationAgent(settings)
            try:
                with patch("app.providers.OpenAICompatibleProvider.generate", return_value=SimpleNamespace(content="smoke ok", prompt_tokens=12, completion_tokens=7, raw={"id": "x"})):
                    reply = agent.ask_sync("hello smoke")
                self.assertEqual(reply.content, "smoke ok")
                exported = agent.export_conversation()
                self.assertEqual(exported["conversation"]["user_id"], "smoke-user")
                self.assertTrue(any(item["role"] == "assistant" for item in exported["messages"]))
                integrity = agent._store.check_integrity(agent.state.conversation_id)
                self.assertTrue(integrity["valid"])
            finally:
                agent.close()


if __name__ == "__main__":
    unittest.main()
