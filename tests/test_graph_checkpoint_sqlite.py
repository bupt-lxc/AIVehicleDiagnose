from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    from app.graph_pipeline import DiagnosticGraph
except Exception:
    DiagnosticGraph = None


class _FakeLLM:
    def __init__(self, fail_once: bool = False) -> None:
        self._fail_once = fail_once
        self._invocations = 0

    def invoke(self, prompt: str) -> SimpleNamespace:
        self._invocations += 1
        if self._fail_once and self._invocations == 1:
            raise RuntimeError("forced llm failure")
        return SimpleNamespace(content=f"analysis::{len(prompt)}")


class _FakeLLMManager:
    def __init__(self, fail_once: bool = False) -> None:
        self.active_profile_name = "test-profile"
        self._llm = _FakeLLM(fail_once=fail_once)

    def get_llm(self, temperature: float = 0.1) -> _FakeLLM:
        return self._llm


class _FakeRAGClient:
    def __init__(self, fail_once: bool = False) -> None:
        self._fail_once = fail_once
        self._calls = 0

    def query_sync(self, question: str, top_k: int = 5, collection: str | None = None) -> dict[str, object]:
        self._calls += 1
        if self._fail_once and self._calls == 1:
            raise RuntimeError("forced rag failure")
        return {
            "content": f"context::{question}::{top_k}::{collection or 'default'}",
            "citations": [{"id": "doc-1", "score": 0.9}],
            "is_error": False,
        }


@unittest.skipIf(DiagnosticGraph is None, "langgraph runtime is not available")
class GraphCheckpointSqliteTests(unittest.TestCase):
    def test_persist_full_node_events_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "graph.sqlite"
            graph = DiagnosticGraph(
                llm_manager=_FakeLLMManager(),
                rag_client=_FakeRAGClient(),
                checkpoint_db_path=str(db_path),
                audit_flush_every=1,
                payload_limit_bytes=1024 * 16,
                node_history_limit=1000,
            )
            try:
                result = graph.run(
                    question="奥迪销量",
                    top_k=3,
                    collection="default",
                    thread_id="thread-A",
                    metadata={"source": "unit-test"},
                )
                self.assertEqual(result["thread_id"], "thread-A")
                self.assertEqual(result["question"], "奥迪销量")
                self.assertIn("analysis::", result["analysis"])
                run_id = result["run_id"]
                trace = graph.get_run_trace(run_id)
                self.assertEqual(trace["run"]["status"], "succeeded")
                self.assertGreaterEqual(len(trace["node_events"]), 6)
                self.assertTrue(any(event["node_id"] == "retrieve" for event in trace["node_events"]))
                self.assertTrue(any(event["phase"] == "succeeded" for event in trace["node_events"]))
                replayed = graph.replay_state(run_id)
                self.assertEqual(replayed["question"], "奥迪销量")
                self.assertIn("analysis", replayed)
                history = graph.get_execution_history(thread_id="thread-A", limit=5)
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0]["run_id"], run_id)
            finally:
                graph.close()

    def test_resume_after_interruption_from_specific_node(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "graph.sqlite"
            rag = _FakeRAGClient(fail_once=True)
            graph = DiagnosticGraph(
                llm_manager=_FakeLLMManager(),
                rag_client=rag,
                checkpoint_db_path=str(db_path),
                audit_flush_every=1,
                payload_limit_bytes=1024 * 16,
                node_history_limit=1000,
            )
            try:
                with self.assertRaises(RuntimeError):
                    graph.run(
                        question="中断恢复测试",
                        top_k=2,
                        collection=None,
                        thread_id="thread-B",
                        metadata={"phase": "first-run"},
                    )
                resumed = graph.resume_from_node(
                    thread_id="thread-B",
                    node_id="retrieve",
                    state_override={"question": "中断恢复测试", "top_k": 2, "collection": None},
                    metadata={"phase": "resume"},
                )
                self.assertEqual(resumed["thread_id"], "thread-B")
                self.assertIn("analysis::", resumed["analysis"])
                history = graph.get_execution_history(thread_id="thread-B", limit=10)
                self.assertGreaterEqual(len(history), 2)
                self.assertEqual(history[0]["status"], "succeeded")
                self.assertTrue(any(item["status"] == "failed" for item in history))
            finally:
                graph.close()


if __name__ == "__main__":
    unittest.main()
