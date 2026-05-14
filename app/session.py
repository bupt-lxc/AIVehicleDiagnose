from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import uuid
import weakref
import atexit
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .compression import CompressionInput, CompressionRegistry, estimate_tokens
from .observability import AuditContext, AuditLogger


@dataclass(slots=True)
class ConversationState:
    conversation_id: str
    user_id: str
    session_id: str
    max_context_tokens: int
    compression_strategy: str
    summary_memory: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    turn_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "max_context_tokens": self.max_context_tokens,
            "compression_strategy": self.compression_strategy,
            "summary_memory": self.summary_memory,
            "messages": self.messages,
            "turn_index": self.turn_index,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConversationState":
        conversation_id = str(payload.get("conversation_id", "")).strip() or f"conv_{uuid.uuid4().hex}"
        return cls(
            conversation_id=conversation_id,
            user_id=str(payload.get("user_id", "default-user")),
            session_id=str(payload.get("session_id", "default-session")),
            max_context_tokens=int(payload.get("max_context_tokens", 30000)),
            compression_strategy=str(payload.get("compression_strategy", "summary")),
            summary_memory=str(payload.get("summary_memory", "")),
            messages=list(payload.get("messages", [])),
            turn_index=int(payload.get("turn_index", 0)),
        )


class ConversationStore:
    _instances = weakref.WeakSet()

    @classmethod
    def _cleanup_all(cls):
        for instance in list(cls._instances):
            try:
                instance.close()
            except Exception:
                pass

    def __init__(self, root_dir: Path, vector_collection: Any | None = None, collection_name: str = "conversation_history") -> None:
        self._root_dir = root_dir
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._multimodal_dir = self._root_dir / "multimodal"
        self._multimodal_dir.mkdir(parents=True, exist_ok=True)
        self._markdown_dir = self._multimodal_dir / "markdown"
        self._markdown_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._multimodal_dir / "conversation_multimodal.sqlite"
        
        self._write_conn = None
        self._read_conn = None
        self._closed = True
        self._write_lock = threading.RLock()
        self._vector_client: Any | None = None
        self._collection_name = collection_name
        self._vector_collection = None
        
        try:
            self._write_conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._write_conn.row_factory = sqlite3.Row
            self._read_conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._read_conn.row_factory = sqlite3.Row
            self._closed = False
            
            self._vector_collection = vector_collection or self._build_default_vector_collection()
            self._setup_sqlite()
            self._rehydrate_vector_index_if_needed()
            ConversationStore._instances.add(self)
        except Exception:
            self.close()
            raise
    
    def state_path(self, user_id: str, session_id: str) -> Path:
        folder = self._root_dir / user_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{session_id}.state.json"

    def close(self) -> None:
        """Close database connections."""
        with self._write_lock:
            if not self._closed:
                if self._write_conn:
                    self._write_conn.close()
                if self._read_conn:
                    self._read_conn.close()
                self._closed = True

    def __enter__(self) -> "ConversationStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Last resort cleanup
        if hasattr(self, "_closed") and not self._closed:
            self.close()

    def history_path(self, user_id: str, session_id: str) -> Path:
        folder = self._root_dir / user_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{session_id}.history.jsonl"

    def load_or_create(
        self,
        user_id: str,
        session_id: str,
        max_context_tokens: int,
        compression_strategy: str,
    ) -> ConversationState:
        path = self.state_path(user_id, session_id)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            state = ConversationState.from_dict(payload)
            if not state.messages:
                state.messages = self._load_messages_from_sqlite(state.conversation_id)
            self.save_state(state)
            self._remove_legacy_history_file(state.user_id, state.session_id)
            self._ensure_conversation_entry(state)
            return state
        state = ConversationState(
            conversation_id=f"conv_{uuid.uuid4().hex}",
            user_id=user_id,
            session_id=session_id,
            max_context_tokens=max_context_tokens,
            compression_strategy=compression_strategy,
        )
        self.save_state(state)
        self._remove_legacy_history_file(state.user_id, state.session_id)
        self._ensure_conversation_entry(state)
        return state

    def save_state(self, state: ConversationState) -> None:
        path = self.state_path(state.user_id, state.session_id)
        # RES-002: 使用原子写操作
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(
                {
                    "conversation_id": state.conversation_id,
                    "user_id": state.user_id,
                    "session_id": state.session_id,
                    "max_context_tokens": state.max_context_tokens,
                    "compression_strategy": state.compression_strategy,
                    "summary_memory": state.summary_memory,
                    "turn_index": state.turn_index,
                    "message_count": len(state.messages),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temp_path.replace(path)
        self._ensure_conversation_entry(state)

    def _load_messages_from_sqlite(self, conversation_id: str) -> list[dict[str, str]]:
        with self._write_lock:
            rows = self._read_conn.execute(
                "SELECT role, raw_text, content FROM messages WHERE conversation_id = ? ORDER BY turn_index, created_at",
                (conversation_id,),
            ).fetchall()
        restored: list[dict[str, str]] = []
        for row in rows:
            role = str(row["role"])
            if role not in {"user", "assistant", "tool"}:
                continue
            text = str(row["raw_text"] or row["content"] or "").strip()
            if not text:
                continue
            restored.append({"role": role, "content": text})
        return restored

    def _remove_legacy_history_file(self, user_id: str, session_id: str) -> None:
        path = self.history_path(user_id, session_id)
        if path.exists():
            path.unlink()

    def record_turn(
        self,
        *,
        state: ConversationState,
        trace_id: str,
        system_prompt: str,
        system_prompt_version: str,
        user_text: str,
        assistant_text: str,
        prompt_messages: list[dict[str, str]],
        tool_calls: list[dict[str, Any]],
        processing_steps: list[dict[str, Any]],
        assistant_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        created_at = self._utcnow()
        self._ensure_conversation_entry(
            state,
            system_prompt=system_prompt,
            system_prompt_version=system_prompt_version,
        )
        system_message_id = f"msg_{uuid.uuid4().hex}"
        user_message_id = f"msg_{uuid.uuid4().hex}"
        assistant_message_id = f"msg_{uuid.uuid4().hex}"
        normalized_tools: list[dict[str, Any]] = []
        for item in tool_calls:
            request_payload = dict(item.get("request", {}))
            result_payload = dict(item.get("result", {}))
            tool_name = str(item.get("tool_name", "")).strip() or "unknown_tool"
            tool_call_id = str(item.get("tool_call_id", "")).strip() or f"tool_{uuid.uuid4().hex}"
            track_id = str(item.get("track_id", "")).strip() or f"trk_{uuid.uuid4().hex}"
            normalized_tools.append(
                {
                    "tool_call_id": tool_call_id,
                    "track_id": track_id,
                    "round": int(item.get("round", 1) or 1),
                    "tool_name": tool_name,
                    "request": request_payload,
                    "result": result_payload,
                    "status": "ok" if not result_payload.get("is_error") else "error",
                    "started_at": str(item.get("started_at", created_at)),
                    "finished_at": str(item.get("finished_at", self._utcnow())),
                    "error": str(item.get("error", "")),
                }
            )
        normalized_steps: list[dict[str, Any]] = []
        for step in processing_steps:
            normalized_steps.append(
                {
                    "step_id": str(step.get("step_id", "")).strip() or f"step_{uuid.uuid4().hex}",
                    "step_type": str(step.get("step_type", "processing")).strip() or "processing",
                    "input": step.get("input", {}),
                    "output": step.get("output", {}),
                    "metadata": step.get("metadata", {}),
                    "created_at": str(step.get("created_at", self._utcnow())),
                }
            )
        message_rows = [
            self._build_message_row(
                message_id=system_message_id,
                conversation_id=state.conversation_id,
                turn_index=state.turn_index,
                role="system",
                content=system_prompt,
                raw_text=system_prompt,
                created_at=created_at,
                metadata={"version": system_prompt_version},
            ),
            self._build_message_row(
                message_id=user_message_id,
                conversation_id=state.conversation_id,
                turn_index=state.turn_index,
                role="user",
                content=user_text,
                raw_text=user_text,
                created_at=created_at,
                metadata={"trace_id": trace_id},
            ),
            self._build_message_row(
                message_id=assistant_message_id,
                conversation_id=state.conversation_id,
                turn_index=state.turn_index,
                role="assistant",
                content=assistant_text,
                raw_text=assistant_text,
                created_at=created_at,
                metadata=assistant_metadata,
            ),
        ]
        for tool in normalized_tools:
            tool_msg_id = f"msg_{uuid.uuid4().hex}"
            message_rows.append(
                self._build_message_row(
                    message_id=tool_msg_id,
                    conversation_id=state.conversation_id,
                    turn_index=state.turn_index,
                    role="tool",
                    content=json.dumps(tool["result"], ensure_ascii=False),
                    raw_text=str(tool["result"].get("content", "")),
                    created_at=tool["finished_at"],
                    metadata={"tool_call_id": tool["tool_call_id"], "track_id": tool["track_id"], "tool_name": tool["tool_name"]},
                )
            )
            tool["message_id"] = tool_msg_id
        markdown_snapshot = self._render_markdown_turn(
            conversation_id=state.conversation_id,
            trace_id=trace_id,
            turn_index=state.turn_index,
            created_at=created_at,
            system_prompt=system_prompt,
            system_prompt_version=system_prompt_version,
            user_text=user_text,
            assistant_text=assistant_text,
            prompt_messages=prompt_messages,
            tool_calls=normalized_tools,
            processing_steps=normalized_steps,
            assistant_metadata=assistant_metadata,
        )
        vector_docs = self._build_vector_documents(
            conversation_id=state.conversation_id,
            trace_id=trace_id,
            created_at=created_at,
            message_rows=message_rows,
            tool_calls=normalized_tools,
            processing_steps=normalized_steps,
        )
        markdown_path = self._markdown_path(state.conversation_id)
        previous_markdown = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else ""
        self._append_markdown(markdown_path, markdown_snapshot)
        vector_ids = [item["id"] for item in vector_docs]
        self._vector_collection.upsert(
            ids=vector_ids,
            documents=[item["document"] for item in vector_docs],
            metadatas=[item["metadata"] for item in vector_docs],
        )
        try:
            self._insert_turn_sqlite(
                state=state,
                trace_id=trace_id,
                created_at=created_at,
                message_rows=message_rows,
                tool_calls=normalized_tools,
                processing_steps=normalized_steps,
                assistant_message_id=assistant_message_id,
                assistant_text=assistant_text,
                assistant_metadata=assistant_metadata,
            )
        except Exception:
            markdown_path.write_text(previous_markdown, encoding="utf-8")
            self._vector_collection.delete(ids=vector_ids)
            raise
        return {
            "conversation_id": state.conversation_id,
            "message_ids": [item["message_id"] for item in message_rows],
            "tool_call_ids": [item["tool_call_id"] for item in normalized_tools],
        }

    def query_messages(
        self,
        *,
        conversation_id: str,
        role: str | None = None,
        keyword: str | None = None,
        tool_name: str | None = None,
        start_ts: str | None = None,
        end_ts: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = [
            "SELECT m.*, t.tool_name AS joined_tool_name, t.track_id AS joined_track_id",
            "FROM messages m",
            "LEFT JOIN tool_calls t ON t.message_id = m.message_id",
            "WHERE m.conversation_id = ?",
        ]
        params: list[Any] = [conversation_id]
        if role:
            sql.append("AND m.role = ?")
            params.append(role)
        if keyword:
            sql.append("AND (m.content LIKE ? OR m.raw_text LIKE ?)")
            like_keyword = f"%{keyword}%"
            params.append(like_keyword)
            params.append(like_keyword)
        if tool_name:
            sql.append("AND t.tool_name = ?")
            params.append(tool_name)
        if start_ts:
            sql.append("AND m.created_at >= ?")
            params.append(start_ts)
        if end_ts:
            sql.append("AND m.created_at <= ?")
            params.append(end_ts)
        sql.append("ORDER BY m.turn_index ASC, m.created_at ASC LIMIT ?")
        params.append(max(1, int(limit)))
        rows = self._read_conn.execute("\n".join(sql), params).fetchall()
        return [self._row_to_dict(item) for item in rows]

    def semantic_search(self, query: str, *, conversation_id: str | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        if conversation_id:
            result = self._vector_collection.query(
                query_texts=[query],
                n_results=max(1, int(top_k)),
                where={"conversation_id": conversation_id},
            )
        else:
            result = self._vector_collection.query(
                query_texts=[query],
                n_results=max(1, int(top_k)),
            )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        merged: list[dict[str, Any]] = []
        for idx, item_id in enumerate(ids):
            merged.append(
                {
                    "id": item_id,
                    "document": docs[idx] if idx < len(docs) else "",
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "distance": distances[idx] if idx < len(distances) else 0.0,
                }
            )
        return merged

    def export_conversation(self, conversation_id: str) -> dict[str, Any]:
        conversation = self._read_conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if not conversation:
            raise ValueError(f"conversation_id not found: {conversation_id}")
        markdown_content = ""
        path = self._markdown_path(conversation_id)
        if path.exists():
            markdown_content = path.read_text(encoding="utf-8")
        vector_rows = self._vector_collection.get(where={"conversation_id": conversation_id})
        return {
            "conversation": self._row_to_dict(conversation),
            "messages": [self._row_to_dict(item) for item in self._read_conn.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY turn_index, created_at", (conversation_id,)).fetchall()],
            "tool_calls": [self._row_to_dict(item) for item in self._read_conn.execute("SELECT * FROM tool_calls WHERE conversation_id = ? ORDER BY started_at", (conversation_id,)).fetchall()],
            "processing_steps": [self._row_to_dict(item) for item in self._read_conn.execute("SELECT * FROM processing_steps WHERE conversation_id = ? ORDER BY created_at", (conversation_id,)).fetchall()],
            "final_outputs": [self._row_to_dict(item) for item in self._read_conn.execute("SELECT * FROM final_outputs WHERE conversation_id = ? ORDER BY created_at", (conversation_id,)).fetchall()],
            "markdown": markdown_content,
            "vectors": vector_rows,
        }

    def delete_conversation(self, conversation_id: str) -> None:
        with self._write_lock:
            with self._write_conn:
                self._write_conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
            path = self._markdown_path(conversation_id)
            if path.exists():
                path.unlink()
            vector_rows = self._vector_collection.get(where={"conversation_id": conversation_id})
            vector_ids = vector_rows.get("ids", [])
            if vector_ids:
                self._vector_collection.delete(ids=vector_ids)

    def check_integrity(self, conversation_id: str) -> dict[str, Any]:
        message_count = int(
            self._read_conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)).fetchone()[0]
        )
        tool_count = int(
            self._read_conn.execute("SELECT COUNT(*) FROM tool_calls WHERE conversation_id = ?", (conversation_id,)).fetchone()[0]
        )
        final_count = int(
            self._read_conn.execute("SELECT COUNT(*) FROM final_outputs WHERE conversation_id = ?", (conversation_id,)).fetchone()[0]
        )
        markdown_path = self._markdown_path(conversation_id)
        markdown_exists = markdown_path.exists()
        markdown_message_markers = 0
        if markdown_exists:
            markdown_message_markers = markdown_path.read_text(encoding="utf-8").count("### Message ")
        vector_rows = self._vector_collection.get(where={"conversation_id": conversation_id})
        vector_count = len(vector_rows.get("ids", []))
        valid = markdown_exists and message_count >= 2 and final_count >= 1 and vector_count >= message_count
        return {
            "conversation_id": conversation_id,
            "message_count": message_count,
            "tool_call_count": tool_count,
            "final_output_count": final_count,
            "markdown_exists": markdown_exists,
            "markdown_message_markers": markdown_message_markers,
            "vector_count": vector_count,
            "valid": valid,
        }

    def recover_conversation(self, conversation_id: str) -> dict[str, Any]:
        rows = [self._row_to_dict(item) for item in self._read_conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY turn_index, created_at",
            (conversation_id,),
        ).fetchall()]
        if not rows:
            raise ValueError(f"conversation_id not found or empty: {conversation_id}")
        self._markdown_path(conversation_id).write_text(self._render_markdown_from_rows(conversation_id, rows), encoding="utf-8")
        vector_rows = self._vector_collection.get(where={"conversation_id": conversation_id})
        vector_ids = vector_rows.get("ids", [])
        if vector_ids:
            self._vector_collection.delete(ids=vector_ids)
        docs = [
            {
                "id": f"vec_{item['message_id']}",
                "document": f"[{item['role']}] {item['raw_text'] or item['content']}",
                "metadata": {
                    "conversation_id": conversation_id,
                    "message_id": item["message_id"],
                    "role": item["role"],
                    "turn_index": item["turn_index"],
                    "created_at": item["created_at"],
                },
            }
            for item in rows
        ]
        self._vector_collection.upsert(
            ids=[item["id"] for item in docs],
            documents=[item["document"] for item in docs],
            metadatas=[item["metadata"] for item in docs],
        )
        return self.check_integrity(conversation_id)

    def rebuild_markdown(self, conversation_id: str) -> dict[str, Any]:
        conversation_row = self._read_conn.execute(
            "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if not conversation_row:
            raise ValueError(f"conversation_id not found: {conversation_id}")
        message_rows = [self._row_to_dict(item) for item in self._read_conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY turn_index, created_at",
            (conversation_id,),
        ).fetchall()]
        if not message_rows:
            raise ValueError(f"conversation_id not found or empty: {conversation_id}")
        tool_rows = [self._row_to_dict(item) for item in self._read_conn.execute(
            "SELECT * FROM tool_calls WHERE conversation_id = ? ORDER BY started_at",
            (conversation_id,),
        ).fetchall()]
        step_rows = [self._row_to_dict(item) for item in self._read_conn.execute(
            "SELECT * FROM processing_steps WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        ).fetchall()]
        final_rows = [self._row_to_dict(item) for item in self._read_conn.execute(
            "SELECT * FROM final_outputs WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        ).fetchall()]
        message_by_id = {str(item["message_id"]): item for item in message_rows}
        final_by_message_id = {str(item["message_id"]): item for item in final_rows}
        grouped: dict[int, list[dict[str, Any]]] = {}
        for item in message_rows:
            grouped.setdefault(int(item["turn_index"]), []).append(item)
        snapshots: list[str] = []
        for turn_index in sorted(grouped):
            turn_messages = grouped[turn_index]
            assistant_msg = next((item for item in turn_messages if item.get("role") == "assistant"), None)
            user_msg = next((item for item in turn_messages if item.get("role") == "user"), None)
            system_msg = next((item for item in turn_messages if item.get("role") == "system"), None)
            if not assistant_msg:
                continue
            assistant_message_id = str(assistant_msg.get("message_id", ""))
            turn_steps = [item for item in step_rows if str(item.get("message_id", "")) == assistant_message_id]
            round_by_tool_call_id: dict[str, int] = {}
            for step in turn_steps:
                if str(step.get("step_type", "")) != "llm_round":
                    continue
                step_input = step.get("input_json", {}) if isinstance(step.get("input_json"), dict) else {}
                step_output = step.get("output_json", {}) if isinstance(step.get("output_json"), dict) else {}
                round_no = int(step_input.get("round", 1) or 1)
                tool_calls = step_output.get("tool_calls", [])
                if not isinstance(tool_calls, list):
                    continue
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    tool_call_id = str(tool_call.get("id", "")).strip()
                    if tool_call_id:
                        round_by_tool_call_id[tool_call_id] = round_no
            turn_tools: list[dict[str, Any]] = []
            for item in tool_rows:
                linked_message = message_by_id.get(str(item.get("message_id", "")), {})
                if int(linked_message.get("turn_index", -1)) != turn_index:
                    continue
                turn_tools.append(
                    {
                        "tool_call_id": item["tool_call_id"],
                        "track_id": item["track_id"],
                        "round": round_by_tool_call_id.get(str(item["tool_call_id"]), 1),
                        "tool_name": item["tool_name"],
                        "request": item.get("request_json", {}),
                        "result": item.get("result_json", {}),
                        "status": item.get("status", "ok"),
                        "started_at": item.get("started_at", ""),
                        "finished_at": item.get("finished_at", ""),
                        "error": item.get("error_text", ""),
                    }
                )
            prompt_messages: list[dict[str, str]] = []
            prompt_step = next((item for item in turn_steps if str(item.get("step_type", "")) == "prompt_assembled"), None)
            if prompt_step:
                prompt_output = prompt_step.get("output_json", {})
                if isinstance(prompt_output, dict):
                    loaded_messages = prompt_output.get("prompt_messages")
                    if isinstance(loaded_messages, list):
                        prompt_messages = [msg for msg in loaded_messages if isinstance(msg, dict)]
            if not prompt_messages:
                prompt_messages = [
                    {"role": "system", "content": str(system_msg.get("raw_text", "") if system_msg else "")},
                    {"role": "user", "content": str(user_msg.get("raw_text", "") if user_msg else "")},
                ]
            final_row = final_by_message_id.get(assistant_message_id, {})
            metadata = final_row.get("metadata_json", {})
            assistant_metadata = metadata if isinstance(metadata, dict) else {}
            trace_id = str(assistant_metadata.get("trace_id", "")).strip()
            if not trace_id and user_msg:
                user_meta = user_msg.get("metadata_json", {})
                if isinstance(user_meta, dict):
                    trace_id = str(user_meta.get("trace_id", "")).strip()
            if not trace_id:
                trace_id = f"trace_turn_{turn_index}"
            system_metadata: dict[str, Any] = {}
            if system_msg:
                raw_system_metadata = system_msg.get("metadata_json", {})
                if isinstance(raw_system_metadata, dict):
                    system_metadata = raw_system_metadata
            snapshots.append(
                self._render_markdown_turn(
                    conversation_id=conversation_id,
                    trace_id=trace_id,
                    turn_index=turn_index,
                    created_at=str(assistant_msg.get("created_at", self._utcnow())),
                    system_prompt=str(system_msg.get("raw_text", "") if system_msg else ""),
                    system_prompt_version=str(system_metadata.get("version", "v1")),
                    user_text=str(user_msg.get("raw_text", "") if user_msg else ""),
                    assistant_text=str(assistant_msg.get("raw_text", "") if assistant_msg else ""),
                    prompt_messages=prompt_messages,
                    tool_calls=turn_tools,
                    processing_steps=[
                        {
                            "step_id": item.get("step_id", ""),
                            "step_type": item.get("step_type", "processing"),
                            "input": item.get("input_json", {}) if isinstance(item.get("input_json"), dict) else {},
                            "output": item.get("output_json", {}) if isinstance(item.get("output_json"), dict) else {},
                            "metadata": item.get("metadata_json", {}) if isinstance(item.get("metadata_json"), dict) else {},
                            "created_at": item.get("created_at", ""),
                        }
                        for item in turn_steps
                    ],
                    assistant_metadata=assistant_metadata,
                )
            )
        markdown_path = self._markdown_path(conversation_id)
        markdown_content = f"# Conversation {conversation_id}\n\n" + "".join(snapshots).strip() + "\n"
        markdown_path.write_text(markdown_content, encoding="utf-8")
        return {
            "conversation_id": conversation_id,
            "turn_count": len(snapshots),
            "path": str(markdown_path),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._write_conn.close()
        finally:
            self._read_conn.close()
            if self._vector_client is not None:
                close_fn = getattr(self._vector_client, "close", None)
                if callable(close_fn):
                    close_fn()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return

    def _utcnow(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _setup_sqlite(self) -> None:
        with self._write_conn:
            self._write_conn.execute("PRAGMA journal_mode=WAL")
            self._write_conn.execute("PRAGMA synchronous=NORMAL")
            self._write_conn.execute("PRAGMA foreign_keys=ON")
            self._write_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    system_prompt TEXT NOT NULL DEFAULT '',
                    system_prompt_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    checksum TEXT NOT NULL
                )
                """
            )
            self._write_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    checksum TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                )
                """
            )
            self._write_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_calls (
                    tool_call_id TEXT PRIMARY KEY,
                    track_id TEXT NOT NULL UNIQUE,
                    conversation_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    error_text TEXT NOT NULL DEFAULT '',
                    checksum TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    FOREIGN KEY(message_id) REFERENCES messages(message_id) ON DELETE CASCADE
                )
                """
            )
            self._write_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_steps (
                    step_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    step_type TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    FOREIGN KEY(message_id) REFERENCES messages(message_id) ON DELETE CASCADE
                )
                """
            )
            self._write_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS final_outputs (
                    output_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    FOREIGN KEY(message_id) REFERENCES messages(message_id) ON DELETE CASCADE
                )
                """
            )
            self._write_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation_turn ON messages(conversation_id, turn_index, created_at)"
            )
            self._write_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_role_content ON messages(conversation_id, role, created_at)"
            )
            self._write_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tool_calls_conversation_tool ON tool_calls(conversation_id, tool_name, started_at)"
            )
            self._write_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_processing_steps_conversation_type ON processing_steps(conversation_id, step_type, created_at)"
            )
        self._read_conn.execute("PRAGMA foreign_keys=ON")

    def _build_default_vector_collection(self) -> Any:
        try:
            import chromadb
        except Exception:
            return _InMemoryVectorCollection()
        chroma_path = self._multimodal_dir / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_path))
        self._vector_client = client
        return client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=_HashEmbeddingFunction(),
        )

    def _rehydrate_vector_index_if_needed(self) -> None:
        if not isinstance(self._vector_collection, _InMemoryVectorCollection):
            return
        message_rows = self._read_conn.execute(
            "SELECT message_id, conversation_id, role, turn_index, content, raw_text, created_at FROM messages ORDER BY created_at"
        ).fetchall()
        tool_rows = self._read_conn.execute(
            "SELECT tool_call_id, conversation_id, tool_name, track_id, request_json, result_json, finished_at FROM tool_calls ORDER BY finished_at"
        ).fetchall()
        step_rows = self._read_conn.execute(
            "SELECT step_id, conversation_id, step_type, input_json, output_json, metadata_json, created_at FROM processing_steps ORDER BY created_at"
        ).fetchall()
        trace_rows = self._read_conn.execute(
            "SELECT output_id, conversation_id, metadata_json, created_at FROM final_outputs ORDER BY created_at"
        ).fetchall()
        if not message_rows and not tool_rows and not step_rows and not trace_rows:
            return
        docs = []
        for item in message_rows:
            docs.append(
                {
                    "id": f"vec_{item['message_id']}",
                    "document": f"[{item['role']}] {item['raw_text'] or item['content']}",
                    "metadata": {
                        "conversation_id": item["conversation_id"],
                        "message_id": item["message_id"],
                        "role": item["role"],
                        "turn_index": item["turn_index"],
                        "created_at": item["created_at"],
                    },
                }
            )
        for item in tool_rows:
            docs.append(
                {
                    "id": f"vec_{item['tool_call_id']}",
                    "document": json.dumps(
                        {
                            "request": json.loads(item["request_json"]) if item["request_json"] else {},
                            "result": json.loads(item["result_json"]) if item["result_json"] else {},
                        },
                        ensure_ascii=False,
                    ),
                    "metadata": {
                        "conversation_id": item["conversation_id"],
                        "tool_call_id": item["tool_call_id"],
                        "track_id": item["track_id"],
                        "tool_name": item["tool_name"],
                        "created_at": item["finished_at"],
                    },
                }
            )
        for item in step_rows:
            docs.append(
                {
                    "id": f"vec_{item['step_id']}",
                    "document": json.dumps(
                        {
                            "input": json.loads(item["input_json"]) if item["input_json"] else {},
                            "output": json.loads(item["output_json"]) if item["output_json"] else {},
                            "metadata": json.loads(item["metadata_json"]) if item["metadata_json"] else {},
                        },
                        ensure_ascii=False,
                    ),
                    "metadata": {
                        "conversation_id": item["conversation_id"],
                        "step_id": item["step_id"],
                        "step_type": item["step_type"],
                        "created_at": item["created_at"],
                    },
                }
            )
        for item in trace_rows:
            metadata = json.loads(item["metadata_json"]) if item["metadata_json"] else {}
            trace_id = str(metadata.get("trace_id", "")).strip()
            if not trace_id:
                continue
            docs.append(
                {
                    "id": f"vec_trace_{trace_id}",
                    "document": f"trace_id={trace_id} created_at={item['created_at']}",
                    "metadata": {
                        "conversation_id": item["conversation_id"],
                        "trace_id": trace_id,
                        "created_at": item["created_at"],
                        "record_type": "trace",
                    },
                }
            )
        self._vector_collection.upsert(
            ids=[item["id"] for item in docs],
            documents=[item["document"] for item in docs],
            metadatas=[item["metadata"] for item in docs],
        )

    def _ensure_conversation_entry(
        self,
        state: ConversationState,
        system_prompt: str = "",
        system_prompt_version: str = "v1",
    ) -> None:
        created_at = self._utcnow()
        payload = {
            "conversation_id": state.conversation_id,
            "user_id": state.user_id,
            "session_id": state.session_id,
            "system_prompt": system_prompt,
            "system_prompt_version": system_prompt_version,
            "updated_at": created_at,
        }
        checksum = self._checksum(payload)
        with self._write_lock:
            with self._write_conn:
                current = self._write_conn.execute(
                    "SELECT created_at FROM conversations WHERE conversation_id = ?",
                    (state.conversation_id,),
                ).fetchone()
                if current:
                    self._write_conn.execute(
                        """
                        UPDATE conversations
                        SET user_id = ?, session_id = ?, system_prompt = CASE WHEN ? = '' THEN system_prompt ELSE ? END,
                            system_prompt_version = CASE WHEN ? = '' THEN system_prompt_version ELSE ? END,
                            updated_at = ?, checksum = ?
                        WHERE conversation_id = ?
                        """,
                        (
                            state.user_id,
                            state.session_id,
                            system_prompt,
                            system_prompt,
                            system_prompt_version,
                            system_prompt_version,
                            created_at,
                            checksum,
                            state.conversation_id,
                        ),
                    )
                else:
                    self._write_conn.execute(
                        """
                        INSERT INTO conversations(
                            conversation_id, user_id, session_id, system_prompt, system_prompt_version,
                            created_at, updated_at, metadata_json, checksum
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            state.conversation_id,
                            state.user_id,
                            state.session_id,
                            system_prompt,
                            system_prompt_version,
                            created_at,
                            created_at,
                            "{}",
                            checksum,
                        ),
                    )

    def _build_message_row(
        self,
        *,
        message_id: str,
        conversation_id: str,
        turn_index: int,
        role: str,
        content: str,
        raw_text: str,
        created_at: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "role": role,
            "content": content,
            "raw_text": raw_text,
            "created_at": created_at,
            "metadata": metadata,
        }
        payload["checksum"] = self._checksum(payload)
        return payload

    def _insert_turn_sqlite(
        self,
        *,
        state: ConversationState,
        trace_id: str,
        created_at: str,
        message_rows: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        processing_steps: list[dict[str, Any]],
        assistant_message_id: str,
        assistant_text: str,
        assistant_metadata: dict[str, Any],
    ) -> None:
        with self._write_lock:
            with self._write_conn:
                self._write_conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                    (created_at, state.conversation_id),
                )
                for row in message_rows:
                    self._write_conn.execute(
                        """
                        INSERT INTO messages(
                            message_id, conversation_id, turn_index, role, content, raw_text, created_at, metadata_json, checksum
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["message_id"],
                            row["conversation_id"],
                            row["turn_index"],
                            row["role"],
                            row["content"],
                            row["raw_text"],
                            row["created_at"],
                            json.dumps(row["metadata"], ensure_ascii=False),
                            row["checksum"],
                        ),
                    )
                for item in tool_calls:
                    payload = {
                        "tool_call_id": item["tool_call_id"],
                        "track_id": item["track_id"],
                        "conversation_id": state.conversation_id,
                        "message_id": item["message_id"],
                        "tool_name": item["tool_name"],
                        "request": item["request"],
                        "result": item["result"],
                        "status": item["status"],
                        "started_at": item["started_at"],
                        "finished_at": item["finished_at"],
                        "error": item["error"],
                    }
                    self._write_conn.execute(
                        """
                        INSERT INTO tool_calls(
                            tool_call_id, track_id, conversation_id, message_id, tool_name,
                            request_json, result_json, status, started_at, finished_at, error_text, checksum
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item["tool_call_id"],
                            item["track_id"],
                            state.conversation_id,
                            item["message_id"],
                            item["tool_name"],
                            json.dumps(item["request"], ensure_ascii=False),
                            json.dumps(item["result"], ensure_ascii=False),
                            item["status"],
                            item["started_at"],
                            item["finished_at"],
                            item["error"],
                            self._checksum(payload),
                        ),
                    )
                for step in processing_steps:
                    payload = {
                        "step_id": step["step_id"],
                        "conversation_id": state.conversation_id,
                        "message_id": assistant_message_id,
                        "step_type": step["step_type"],
                        "input": step["input"],
                        "output": step["output"],
                        "metadata": step["metadata"],
                        "created_at": step["created_at"],
                    }
                    self._write_conn.execute(
                        """
                        INSERT INTO processing_steps(
                            step_id, conversation_id, message_id, step_type,
                            input_json, output_json, metadata_json, created_at, checksum
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            step["step_id"],
                            state.conversation_id,
                            assistant_message_id,
                            step["step_type"],
                            json.dumps(step["input"], ensure_ascii=False),
                            json.dumps(step["output"], ensure_ascii=False),
                            json.dumps(step["metadata"], ensure_ascii=False),
                            step["created_at"],
                            self._checksum(payload),
                        ),
                    )
                final_payload = {
                    "output_id": f"out_{uuid.uuid4().hex}",
                    "conversation_id": state.conversation_id,
                    "message_id": assistant_message_id,
                    "content": assistant_text,
                    "metadata": assistant_metadata | {"trace_id": trace_id},
                    "created_at": self._utcnow(),
                }
                self._write_conn.execute(
                    """
                    INSERT INTO final_outputs(
                        output_id, conversation_id, message_id, content, metadata_json, created_at, checksum
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        final_payload["output_id"],
                        final_payload["conversation_id"],
                        final_payload["message_id"],
                        final_payload["content"],
                        json.dumps(final_payload["metadata"], ensure_ascii=False),
                        final_payload["created_at"],
                        self._checksum(final_payload),
                    ),
                )

    def _render_markdown_turn(
        self,
        *,
        conversation_id: str,
        trace_id: str,
        turn_index: int,
        created_at: str,
        system_prompt: str,
        system_prompt_version: str,
        user_text: str,
        assistant_text: str,
        prompt_messages: list[dict[str, str]],
        tool_calls: list[dict[str, Any]],
        processing_steps: list[dict[str, Any]],
        assistant_metadata: dict[str, Any],
    ) -> str:
        rounds: dict[int, dict[str, Any]] = {}
        for step in processing_steps:
            if str(step.get("step_type", "")) != "llm_round":
                continue
            step_input = step.get("input", {}) if isinstance(step.get("input"), dict) else {}
            step_output = step.get("output", {}) if isinstance(step.get("output"), dict) else {}
            round_no = int(step_input.get("round", 1) or 1)
            rounds[round_no] = {
                "provider": str(step_input.get("provider", "")),
                "finish_reason": str(step_output.get("finish_reason", "")),
                "assistant_preview": str(step_output.get("assistant_preview", "")),
                "tool_calls": step_output.get("tool_calls", []),
            }
        for item in tool_calls:
            round_no = int(item.get("round", 1) or 1)
            rounds.setdefault(
                round_no,
                {
                    "provider": str(assistant_metadata.get("provider", "")),
                    "finish_reason": "",
                    "assistant_preview": "",
                    "tool_calls": [],
                },
            )
        lines = []
        if not self._markdown_path(conversation_id).exists():
            lines.append(f"# Conversation {conversation_id}")
            lines.append("")
        lines.append(f"## Turn {turn_index}")
        lines.append(f"- trace_id: {trace_id}")
        lines.append(f"- created_at: {created_at}")
        lines.append(f"- system_prompt_version: {system_prompt_version}")
        lines.append("")
        lines.append("### Round Timeline")
        lines.append("| Round | Provider | Finish | Tool Calls | Assistant Preview |")
        lines.append("|---:|---|---|---:|---|")
        if rounds:
            for round_no in sorted(rounds):
                item = rounds[round_no]
                tool_count = len([t for t in tool_calls if int(t.get("round", 1) or 1) == round_no])
                preview = str(item.get("assistant_preview", "")).replace("\n", " ").strip()
                if len(preview) > 120:
                    preview = f"{preview[:117]}..."
                lines.append(
                    f"| {round_no} | {item.get('provider', '') or '-'} | "
                    f"{item.get('finish_reason', '') or '-'} | {tool_count} | {preview or '-'} |"
                )
        else:
            lines.append("| 1 | - | - | 0 | - |")
        lines.append("")
        for round_no in sorted(rounds) if rounds else [1]:
            lines.append(f"### Round {round_no}")
            lines.append("")
            lines.append("#### System Prompt")
            lines.append(system_prompt)
            lines.append("")
            lines.append("#### User Prompt")
            lines.append(user_text)
            lines.append("")
            lines.append("#### Tool Calls")
            round_tools = [t for t in tool_calls if int(t.get('round', 1) or 1) == round_no]
            if not round_tools:
                lines.append("- none")
                lines.append("")
            for item in round_tools:
                lines.append(f"##### {item['tool_name']}")
                lines.append(f"- tool_call_id: {item['tool_call_id']}")
                lines.append(f"- track_id: {item['track_id']}")
                lines.append(f"- status: {item['status']}")
                lines.append("- tool_call")
                lines.append("```json")
                lines.append(json.dumps(item["request"], ensure_ascii=False, indent=2))
                lines.append("```")
                lines.append("- tool_call_result")
                lines.append("```json")
                lines.append(json.dumps(item["result"], ensure_ascii=False, indent=2))
                lines.append("```")
                lines.append("")
            round_preview = rounds.get(round_no, {}).get("assistant_preview", "")
            if round_preview:
                lines.append("#### Assistant Draft")
                lines.append(str(round_preview))
                lines.append("")
        lines.append("### Assistant Response")
        lines.append(assistant_text)
        lines.append("")
        lines.append("### Assistant Metadata")
        lines.append("```json")
        lines.append(json.dumps(assistant_metadata, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("### Prompt Messages")
        lines.append("```json")
        lines.append(json.dumps(prompt_messages, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("### Turn Snapshot")
        lines.append("```json")
        lines.append(
            json.dumps(
                {
                    "prompt_message_count": len(prompt_messages),
                    "tool_call_count": len(tool_calls),
                    "processing_step_count": len(processing_steps),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        lines.append("```")
        lines.append("")
        return "\n".join(lines).strip() + "\n\n"

    def _render_markdown_from_rows(self, conversation_id: str, rows: list[dict[str, Any]]) -> str:
        lines = [f"# Conversation {conversation_id}", ""]
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(int(row["turn_index"]), []).append(row)
        for turn in sorted(grouped):
            lines.append(f"## Turn {turn}")
            lines.append("")
            for item in grouped[turn]:
                lines.append(f"### Message {item['role']}")
                lines.append(item.get("raw_text") or item.get("content") or "")
                lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _append_markdown(self, markdown_path: Path, payload: str) -> None:
        if markdown_path.exists():
            current = markdown_path.read_text(encoding="utf-8")
            markdown_path.write_text(current + payload, encoding="utf-8")
            return
        markdown_path.write_text(payload, encoding="utf-8")

    def _build_vector_documents(
        self,
        *,
        conversation_id: str,
        trace_id: str,
        created_at: str,
        message_rows: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        processing_steps: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for row in message_rows:
            docs.append(
                {
                    "id": f"vec_{row['message_id']}",
                    "document": f"[{row['role']}] {row['raw_text'] or row['content']}",
                    "metadata": {
                        "conversation_id": conversation_id,
                        "trace_id": trace_id,
                        "message_id": row["message_id"],
                        "role": row["role"],
                        "turn_index": row["turn_index"],
                        "created_at": row["created_at"],
                    },
                }
            )
        for item in tool_calls:
            docs.append(
                {
                    "id": f"vec_{item['tool_call_id']}",
                    "document": json.dumps({"request": item["request"], "result": item["result"]}, ensure_ascii=False),
                    "metadata": {
                        "conversation_id": conversation_id,
                        "trace_id": trace_id,
                        "tool_call_id": item["tool_call_id"],
                        "track_id": item["track_id"],
                        "tool_name": item["tool_name"],
                        "created_at": item["finished_at"],
                    },
                }
            )
        for step in processing_steps:
            docs.append(
                {
                    "id": f"vec_{step['step_id']}",
                    "document": json.dumps(
                        {
                            "input": step["input"],
                            "output": step["output"],
                            "metadata": step["metadata"],
                        },
                        ensure_ascii=False,
                    ),
                    "metadata": {
                        "conversation_id": conversation_id,
                        "trace_id": trace_id,
                        "step_id": step["step_id"],
                        "step_type": step["step_type"],
                        "created_at": step["created_at"],
                    },
                }
            )
        docs.append(
            {
                "id": f"vec_trace_{trace_id}",
                "document": f"trace_id={trace_id} created_at={created_at}",
                "metadata": {
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "created_at": created_at,
                    "record_type": "trace",
                },
            }
        )
        return docs

    def _checksum(self, payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _row_to_dict(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        for key in ("metadata_json", "request_json", "result_json", "input_json", "output_json"):
            if key in data and isinstance(data[key], str):
                try:
                    data[key] = json.loads(data[key])
                except json.JSONDecodeError:
                    pass
        return data

    def _markdown_path(self, conversation_id: str) -> Path:
        return self._markdown_dir / f"{conversation_id}.md"


class _HashEmbeddingFunction:
    def name(self=None) -> str:
        return "hash_embedding_function"

    def is_legacy(self=None) -> bool:
        return False

    def default_space(self=None) -> str:
        return "cosine"

    def supported_spaces(self=None) -> list[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self=None) -> dict[str, Any]:
        return {}

    @classmethod
    def build_from_config(cls, config: dict[str, Any]) -> "_HashEmbeddingFunction":
        return cls()

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str, dim: int = 64) -> list[float]:
        vec = [0.0] * dim
        tokens = [item for item in text.lower().split() if item]
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % dim
            vec[index] += 1.0
        norm = math.sqrt(sum(item * item for item in vec))
        if norm <= 0:
            return vec
        return [item / norm for item in vec]


class _InMemoryVectorCollection:
    def __init__(self) -> None:
        self._embedding = _HashEmbeddingFunction()
        self._docs: dict[str, dict[str, Any]] = {}

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        for idx, item_id in enumerate(ids):
            self._docs[item_id] = {
                "id": item_id,
                "document": documents[idx] if idx < len(documents) else "",
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
            }

    def get(self, where: dict[str, Any] | None = None) -> dict[str, Any]:
        rows = list(self._docs.values())
        if where:
            rows = [item for item in rows if all(item["metadata"].get(key) == value for key, value in where.items())]
        return {
            "ids": [item["id"] for item in rows],
            "documents": [item["document"] for item in rows],
            "metadatas": [item["metadata"] for item in rows],
        }

    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> None:
        if ids:
            for item_id in ids:
                self._docs.pop(item_id, None)
            return
        if where:
            to_delete = [item["id"] for item in self._docs.values() if all(item["metadata"].get(k) == v for k, v in where.items())]
            for item_id in to_delete:
                self._docs.pop(item_id, None)

    def query(self, query_texts: list[str], n_results: int = 5, where: dict[str, Any] | None = None) -> dict[str, Any]:
        query = query_texts[0] if query_texts else ""
        query_vec = self._embedding._embed(query)
        rows = list(self._docs.values())
        if where:
            rows = [item for item in rows if all(item["metadata"].get(k) == v for k, v in where.items())]
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in rows:
            vec = self._embedding._embed(item["document"])
            score = sum(a * b for a, b in zip(query_vec, vec))
            scored.append((score, item))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[: max(1, int(n_results))]
        return {
            "ids": [[item[1]["id"] for item in selected]],
            "documents": [[item[1]["document"] for item in selected]],
            "metadatas": [[item[1]["metadata"] for item in selected]],
            "distances": [[1 - item[0] for item in selected]],
        }


class ContextManager:
    KEY_EVIDENCE_MARKERS = ("DTC", "ECU", "root cause", "根因", "置信度", "tool", "observation")
    def __init__(
        self,
        compression_registry: CompressionRegistry,
        audit_logger: AuditLogger,
    ) -> None:
        self._compression_registry = compression_registry
        self._audit_logger = audit_logger

    def build_prompt_messages(self, state: ConversationState) -> list[dict[str, str]]:
        system_message = {
            "role": "system",
            "content": "You are an AI Agent assistant. Use tools when needed and provide concise helpful answers.",
        }
        prompt_messages: list[dict[str, str]] = [system_message]
        if state.summary_memory.strip():
            prompt_messages.append(
                {
                    "role": "system",
                    "content": f"Compressed context memory: {state.summary_memory}",
                },
            )
        prompt_messages.extend(state.messages)
        return prompt_messages

    def _split_messages_for_rounds(
        self,
        messages: list[dict[str, str]],
        keep_user_rounds: int,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        if not messages:
            return [], []
        user_indices = [idx for idx, item in enumerate(messages) if item.get("role") == "user"]
        if len(user_indices) <= keep_user_rounds:
            return [], messages
        split_index = user_indices[-keep_user_rounds]
        return messages[:split_index], messages[split_index:]

    def maybe_compress(self, state: ConversationState, trace_id: str) -> bool:
        total_text = state.summary_memory + "\n" + "\n".join(item.get("content", "") for item in state.messages)
        total_tokens = estimate_tokens(total_text)
        if total_tokens <= state.max_context_tokens:
            return False
        strategy = self._compression_registry.get(state.compression_strategy)
        old_messages, recent_messages = self._split_messages_for_rounds(
            state.messages,
            keep_user_rounds=3,
        )
        compressed = strategy.compress(
            CompressionInput(
                previous_summary=state.summary_memory,
                old_messages=old_messages,
                max_tokens=max(1000, state.max_context_tokens // 2),
            ),
        )
        evidence_lines = []
        for msg in old_messages[-20:]:
            content = str(msg.get("content", ""))
            if any(marker.lower() in content.lower() for marker in self.KEY_EVIDENCE_MARKERS):
                evidence_lines.append(content[:200])
        preserved_evidence = "\n".join(evidence_lines[:8])
        final_summary = compressed.summary_text
        if preserved_evidence:
            final_summary = f"{compressed.summary_text}\n\nKey evidence retained:\n{preserved_evidence}".strip()
        before_markers = [item for item in self.KEY_EVIDENCE_MARKERS if item.lower() in total_text.lower()]
        after_markers = [item for item in self.KEY_EVIDENCE_MARKERS if item.lower() in final_summary.lower()]
        state.summary_memory = final_summary
        state.messages = recent_messages
        self._audit_logger.record_compression()
        self._audit_logger.emit(
            "context_compressed",
            AuditContext(
                user_id=state.user_id,
                session_id=state.session_id,
                trace_id=trace_id,
                turn_id=state.turn_index,
                metadata={"strategy": compressed.strategy_name},
            ),
            {
                "estimated_tokens": compressed.estimated_tokens,
                "kept_recent_messages": len(recent_messages),
                "compressed_old_messages": len(old_messages),
                "marker_consistency": {
                    "before": before_markers,
                    "after": after_markers,
                    "retention_ratio": round((len(after_markers) / len(before_markers)) if before_markers else 1.0, 4),
                },
            },
        )
        return True


atexit.register(ConversationStore._cleanup_all)
