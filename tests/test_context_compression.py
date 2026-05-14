from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from app.compression import CompressionRegistry
from app.observability import AuditLogger
from app.session import ContextManager, ConversationState


class ContextCompressionTests(unittest.TestCase):
    def test_compress_keeps_recent_three_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = AuditLogger(Path(temp_dir))
            registry = CompressionRegistry()
            manager = ContextManager(registry, logger)
            state = ConversationState(
                conversation_id="conv-c1",
                user_id="u1",
                session_id="s1",
                max_context_tokens=40,
                compression_strategy="summary",
            )
            for idx in range(10):
                state.messages.append({"role": "user", "content": f"user message {idx} " * 8})
                state.messages.append({"role": "assistant", "content": f"assistant message {idx} " * 8})
            compressed = manager.maybe_compress(state, trace_id=str(uuid.uuid4()))
            self.assertTrue(compressed)
            self.assertEqual(len(state.messages), 6)
            self.assertGreater(len(state.summary_memory), 0)


if __name__ == "__main__":
    unittest.main()
