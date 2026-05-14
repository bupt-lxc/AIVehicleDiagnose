from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from app.search_local_history_tool import (
    LocalHistorySearchService,
    SearchLocalHistoryInput,
    build_search_local_history_tool,
    get_tool_definition,
)


class _FakeCollection:
    def __init__(self) -> None:
        self._rows = [
            {
                "id": "vec_1",
                "document": "budget discussion for Q1 plan",
                "metadata": {
                    "conversation_id": "conv_a",
                    "message_id": "msg_1",
                    "trace_id": "trace_1",
                    "created_at": "2026-03-10T10:00:00+00:00",
                },
                "distance": 0.05,
            },
            {
                "id": "vec_2",
                "document": "team standup summary and blockers",
                "metadata": {
                    "conversation_id": "conv_b",
                    "message_id": "msg_2",
                    "trace_id": "trace_2",
                    "created_at": "2026-02-28T09:00:00+00:00",
                },
                "distance": 0.2,
            },
        ]

    def query(self, query_texts, n_results, include):
        rows = self._rows[:n_results]
        return {
            "ids": [[item["id"] for item in rows]],
            "documents": [[item["document"] for item in rows]],
            "metadatas": [[item["metadata"] for item in rows]],
            "distances": [[item["distance"] for item in rows]],
        }

    def get(self, include):
        return {
            "ids": [item["id"] for item in self._rows],
            "documents": [item["document"] for item in self._rows],
            "metadatas": [item["metadata"] for item in self._rows],
        }


class _FakeClient:
    def __init__(self, path: str) -> None:
        self.path = path
        self._collection = _FakeCollection()

    def get_collection(self, name: str):
        return self._collection

    def close(self) -> None:
        return None


class SearchLocalHistoryToolTests(unittest.TestCase):
    def _patch_chromadb(self):
        fake_module = types.SimpleNamespace(PersistentClient=_FakeClient)
        return patch.dict("sys.modules", {"chromadb": fake_module})

    def test_hybrid_search_with_date_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "multimodal" / "chroma").mkdir(parents=True, exist_ok=True)
            service = LocalHistorySearchService(history_root=root)
            payload = SearchLocalHistoryInput(
                query="budget",
                start_date="2026-03-06",
                end_date="2026-03-13",
                mode="hybrid",
                limit=5,
            )
            with self._patch_chromadb():
                result = service.search(payload)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(result["results"]), 1)
            self.assertEqual(result["results"][0]["id"], "vec_1")
            self.assertEqual(result["results"][0]["conversation_id"], "conv_a")

    def test_tool_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "multimodal" / "chroma").mkdir(parents=True, exist_ok=True)
            service = LocalHistorySearchService(history_root=root)
            tool = build_search_local_history_tool(service)
            with self._patch_chromadb():
                with self.assertRaises(Exception):
                    tool.invoke(
                        {
                            "query": "budget",
                            "start_date": "2026-03-20",
                            "end_date": "2026-03-10",
                            "mode": "hybrid",
                            "limit": 5,
                        }
                    )

    def test_get_tool_definition_schema(self) -> None:
        definition = get_tool_definition()
        self.assertEqual(definition["name"], "search_local_history")
        self.assertIn("schema", definition)
        self.assertIn("query", definition["schema"]["properties"])


if __name__ == "__main__":
    unittest.main()
