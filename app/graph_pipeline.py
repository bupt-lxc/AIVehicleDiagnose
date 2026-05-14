from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import sqlite3
import time
import tracemalloc
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Callable
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .llm_manager import LLMManager
from .rag_client import RAGClient


logger = logging.getLogger(__name__)


class DiagnosticState(TypedDict):
    question: str
    top_k: int
    collection: str | None
    resume_from: str | None
    run_id: str
    thread_id: str
    rag_content: str
    citations: list[dict[str, Any]]
    rag_error: bool
    llm_profile: str
    analysis: str


@dataclass(slots=True)
class _RunContext:
    run_id: str
    thread_id: str
    metadata: dict[str, Any]


_ACTIVE_RUN_CONTEXT: contextvars.ContextVar[_RunContext | None] = contextvars.ContextVar(
    "active_run_context",
    default=None,
)


class GraphExecutionStore:
    def __init__(
        self,
        db_path: Path,
        flush_every: int = 8,
        payload_limit_bytes: int = 65536,
        node_history_limit: int = 50000,
    ) -> None:
        self._db_path = db_path
        self._flush_every = max(flush_every, 1)
        self._payload_limit_bytes = max(payload_limit_bytes, 2048)
        self._node_history_limit = max(node_history_limit, 1000)
        self._lock = Lock()
        self._buffer: list[dict[str, Any]] = []
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._prepare_database()

    def _prepare_database(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    input_json TEXT NOT NULL,
                    output_json TEXT,
                    error_stack TEXT,
                    metadata_json TEXT,
                    last_node_id TEXT,
                    node_event_count INTEGER DEFAULT 0,
                    write_count INTEGER DEFAULT 0
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_node_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_ms REAL,
                    input_json TEXT,
                    output_json TEXT,
                    state_json TEXT,
                    exception_stack TEXT,
                    metadata_json TEXT,
                    memory_current_kb INTEGER,
                    memory_peak_kb INTEGER,
                    payload_bytes INTEGER DEFAULT 0
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_runs_thread ON graph_runs(thread_id, started_at DESC)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_node_events_run ON graph_node_events(run_id, started_at ASC)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_node_events_thread ON graph_node_events(thread_id, started_at ASC)"
            )

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _serialize_payload(self, payload: Any) -> tuple[str, int]:
        serialized = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
        size = len(serialized.encode("utf-8"))
        if size <= self._payload_limit_bytes:
            return serialized, size
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        limited = json.dumps(
            {
                "truncated": True,
                "digest": digest,
                "original_bytes": size,
                "preview": serialized[: self._payload_limit_bytes],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return limited, len(limited.encode("utf-8"))

    def start_run(self, run_id: str, thread_id: str, input_payload: dict[str, Any], metadata: dict[str, Any]) -> None:
        input_json, _ = self._serialize_payload(input_payload)
        metadata_json, _ = self._serialize_payload(metadata)
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO graph_runs (
                    run_id, thread_id, status, started_at, input_json, metadata_json
                ) VALUES (?, ?, 'running', ?, ?, ?)
                """,
                (run_id, thread_id, self._now(), input_json, metadata_json),
            )

    def append_node_event(
        self,
        *,
        run_id: str,
        thread_id: str,
        node_id: str,
        phase: str,
        started_at: str,
        finished_at: str | None,
        duration_ms: float | None,
        input_payload: dict[str, Any] | None,
        output_payload: dict[str, Any] | None,
        state_payload: dict[str, Any] | None,
        exception_stack: str | None,
        metadata_payload: dict[str, Any] | None,
        memory_current_kb: int | None,
        memory_peak_kb: int | None,
    ) -> None:
        input_json, input_bytes = self._serialize_payload(input_payload) if input_payload is not None else ("", 0)
        output_json, output_bytes = (
            self._serialize_payload(output_payload) if output_payload is not None else ("", 0)
        )
        state_json, state_bytes = self._serialize_payload(state_payload) if state_payload is not None else ("", 0)
        metadata_json, metadata_bytes = (
            self._serialize_payload(metadata_payload) if metadata_payload is not None else ("", 0)
        )
        payload_bytes = input_bytes + output_bytes + state_bytes + metadata_bytes
        event = {
            "event_id": uuid.uuid4().hex,
            "run_id": run_id,
            "thread_id": thread_id,
            "node_id": node_id,
            "phase": phase,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "input_json": input_json,
            "output_json": output_json,
            "state_json": state_json,
            "exception_stack": exception_stack or "",
            "metadata_json": metadata_json,
            "memory_current_kb": memory_current_kb,
            "memory_peak_kb": memory_peak_kb,
            "payload_bytes": payload_bytes,
        }
        should_flush = False
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self._flush_every:
                should_flush = True
        if should_flush:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            pending = list(self._buffer)
            self._buffer.clear()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.executemany(
                """
                INSERT INTO graph_node_events (
                    event_id, run_id, thread_id, node_id, phase, started_at, finished_at, duration_ms,
                    input_json, output_json, state_json, exception_stack, metadata_json, memory_current_kb,
                    memory_peak_kb, payload_bytes
                ) VALUES (
                    :event_id, :run_id, :thread_id, :node_id, :phase, :started_at, :finished_at, :duration_ms,
                    :input_json, :output_json, :state_json, :exception_stack, :metadata_json, :memory_current_kb,
                    :memory_peak_kb, :payload_bytes
                )
                """,
                pending,
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def finish_run(
        self,
        run_id: str,
        status: str,
        output_payload: dict[str, Any] | None,
        error_stack: str | None,
        last_node_id: str | None,
    ) -> None:
        self.flush()
        output_json, _ = self._serialize_payload(output_payload) if output_payload is not None else ("", 0)
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                """
                UPDATE graph_runs
                SET status = ?,
                    ended_at = ?,
                    output_json = ?,
                    error_stack = ?,
                    last_node_id = ?,
                    node_event_count = (
                        SELECT COUNT(1) FROM graph_node_events WHERE run_id = graph_runs.run_id
                    ),
                    write_count = (
                        SELECT COALESCE(SUM(payload_bytes), 0) FROM graph_node_events WHERE run_id = graph_runs.run_id
                    )
                WHERE run_id = ?
                """,
                (status, self._now(), output_json, error_stack or "", last_node_id, run_id),
            )
            self._conn.execute(
                """
                DELETE FROM graph_node_events
                WHERE rowid IN (
                    SELECT id FROM (
                        SELECT rowid AS id FROM graph_node_events
                        ORDER BY started_at DESC
                        LIMIT -1 OFFSET ?
                    )
                )
                """,
                (self._node_history_limit,),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM graph_runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list_runs(self, thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM graph_runs WHERE thread_id = ? ORDER BY started_at DESC LIMIT ?",
            (thread_id, max(limit, 1)),
        ).fetchall()
        return [self._row_to_dict(item) for item in rows]

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM graph_node_events WHERE run_id = ? ORDER BY started_at ASC",
            (run_id,),
        ).fetchall()
        return [self._row_to_dict(item) for item in rows]

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for key in ("input_json", "output_json", "metadata_json", "state_json"):
            if key in data and data[key]:
                try:
                    data[key] = json.loads(data[key])
                except json.JSONDecodeError:
                    pass
        return data

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._conn.close()


class DiagnosticGraph:
    def __init__(
        self,
        llm_manager: LLMManager,
        rag_client: RAGClient,
        checkpoint_db_path: str = "logs/history/langgraph_checkpoint.sqlite",
        audit_flush_every: int = 8,
        payload_limit_bytes: int = 65536,
        node_history_limit: int = 50000,
    ) -> None:
        self._llm_manager = llm_manager
        self._rag_client = rag_client
        self._checkpoint_db_path = self._resolve_db_path(checkpoint_db_path)
        self._checkpoint_conn = sqlite3.connect(str(self._checkpoint_db_path), check_same_thread=False)
        self._checkpoint_conn.execute("PRAGMA journal_mode=WAL")
        self._checkpoint_conn.execute("PRAGMA synchronous=NORMAL")
        self._checkpointer = SqliteSaver(self._checkpoint_conn)
        self._checkpointer.setup()
        self._execution_store = GraphExecutionStore(
            db_path=self._checkpoint_db_path,
            flush_every=audit_flush_every,
            payload_limit_bytes=payload_limit_bytes,
            node_history_limit=node_history_limit,
        )
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self._graph = self._build_graph()

    def _resolve_db_path(self, db_path: str) -> Path:
        path = Path(db_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _build_graph(self):
        graph = StateGraph(DiagnosticState)
        graph.add_node("route", self._wrap_node("route", self._route_node))
        graph.add_node("retrieve", self._wrap_node("retrieve", self._retrieve_node))
        graph.add_node("analyze", self._wrap_node("analyze", self._analyze_node))
        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", self._choose_next_node, {"retrieve": "retrieve", "analyze": "analyze"})
        graph.add_edge("retrieve", "analyze")
        graph.add_edge("analyze", END)
        return graph.compile(checkpointer=self._checkpointer)

    def _wrap_node(
        self, node_id: str, handler: Callable[[DiagnosticState], DiagnosticState]
    ) -> Callable[[DiagnosticState], DiagnosticState]:
        def wrapped(state: DiagnosticState) -> DiagnosticState:
            run_context = _ACTIVE_RUN_CONTEXT.get()
            started_at = datetime.now(timezone.utc).isoformat()
            perf_start = time.perf_counter()
            current_memory, peak_memory = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
            metadata = {
                "node_id": node_id,
                "llm_profile": self._llm_manager.active_profile_name,
                "run_metadata": run_context.metadata if run_context else {},
            }
            if run_context:
                self._execution_store.append_node_event(
                    run_id=run_context.run_id,
                    thread_id=run_context.thread_id,
                    node_id=node_id,
                    phase="started",
                    started_at=started_at,
                    finished_at=None,
                    duration_ms=None,
                    input_payload=dict(state),
                    output_payload=None,
                    state_payload=None,
                    exception_stack=None,
                    metadata_payload=metadata,
                    memory_current_kb=current_memory // 1024,
                    memory_peak_kb=peak_memory // 1024,
                )
            try:
                output = handler(state)
                duration_ms = (time.perf_counter() - perf_start) * 1000
                current_memory, peak_memory = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
                if run_context:
                    self._execution_store.append_node_event(
                        run_id=run_context.run_id,
                        thread_id=run_context.thread_id,
                        node_id=node_id,
                        phase="succeeded",
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        duration_ms=duration_ms,
                        input_payload=dict(state),
                        output_payload=dict(output),
                        state_payload=dict(output),
                        exception_stack=None,
                        metadata_payload=metadata,
                        memory_current_kb=current_memory // 1024,
                        memory_peak_kb=peak_memory // 1024,
                    )
                return output
            except Exception:
                duration_ms = (time.perf_counter() - perf_start) * 1000
                current_memory, peak_memory = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
                if run_context:
                    self._execution_store.append_node_event(
                        run_id=run_context.run_id,
                        thread_id=run_context.thread_id,
                        node_id=node_id,
                        phase="failed",
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        duration_ms=duration_ms,
                        input_payload=dict(state),
                        output_payload=None,
                        state_payload=dict(state),
                        exception_stack=traceback.format_exc(),
                        metadata_payload=metadata,
                        memory_current_kb=current_memory // 1024,
                        memory_peak_kb=peak_memory // 1024,
                    )
                raise

        return wrapped

    def _route_node(self, state: DiagnosticState) -> DiagnosticState:
        return state

    def _choose_next_node(self, state: DiagnosticState) -> str:
        target = (state.get("resume_from") or "").strip().lower()
        if target in {"retrieve", "analyze"}:
            return target
        return "retrieve"

    def _retrieve_node(self, state: DiagnosticState) -> DiagnosticState:
        question = state["question"]
        top_k = state.get("top_k", 5)
        collection = state.get("collection")
        rag_result = self._rag_client.query_sync(question=question, top_k=top_k, collection=collection)
        logger.info("RAG retrieved: error=%s citations=%s", rag_result["is_error"], len(rag_result["citations"]))
        return {
            **state,
            "rag_content": rag_result["content"],
            "citations": rag_result["citations"],
            "rag_error": rag_result["is_error"],
        }

    def _analyze_node(self, state: DiagnosticState) -> DiagnosticState:
        llm = self._llm_manager.get_llm()
        prompt = (
            "You are a senior diagnostic analyst.\n"
            f"Question:\n{state['question']}\n\n"
            f"Retrieved Context:\n{state['rag_content']}\n\n"
            "Please provide:\n"
            "1) concise conclusion\n"
            "2) key evidence\n"
            "3) risks/uncertainties\n"
            "4) next actions"
        )
        try:
            response = llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            logger.info("LLM analyzed with profile=%s", self._llm_manager.active_profile_name)
        except Exception as exc:
            logger.exception("LLM analyze failed with profile=%s", self._llm_manager.active_profile_name)
            content = (
                "Analysis unavailable due to LLM invocation failure.\n"
                f"Error: {exc}\n"
                "Please verify API quota, model access, and endpoint configuration."
            )
        return {
            **state,
            "analysis": content,
            "llm_profile": self._llm_manager.active_profile_name,
        }

    def _build_default_state(
        self,
        *,
        question: str,
        top_k: int,
        collection: str | None,
        run_id: str,
        thread_id: str,
    ) -> DiagnosticState:
        return {
            "question": question,
            "top_k": top_k,
            "collection": collection,
            "resume_from": None,
            "run_id": run_id,
            "thread_id": thread_id,
            "rag_content": "",
            "citations": [],
            "rag_error": False,
            "llm_profile": self._llm_manager.active_profile_name,
            "analysis": "",
        }

    def _build_config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def run(
        self,
        question: str,
        top_k: int = 5,
        collection: str | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
        resume: bool = False,
        resume_from_node: str | None = None,
        state_override: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_thread_id = (thread_id or "").strip() or f"diagnostic-{uuid.uuid4().hex[:12]}"
        resolved_run_id = (run_id or "").strip() or uuid.uuid4().hex
        config = self._build_config(resolved_thread_id)
        state = self._build_default_state(
            question=question,
            top_k=top_k,
            collection=collection,
            run_id=resolved_run_id,
            thread_id=resolved_thread_id,
        )
        if resume:
            checkpoint_state = self._graph.get_state(config)
            if checkpoint_state and checkpoint_state.values:
                state.update(dict(checkpoint_state.values))
        if state_override:
            state.update(state_override)
        state["question"] = question or state.get("question", "")
        state["top_k"] = int(top_k or state.get("top_k", 5))
        state["collection"] = collection if collection is not None else state.get("collection")
        state["run_id"] = resolved_run_id
        state["thread_id"] = resolved_thread_id
        state["resume_from"] = resume_from_node
        resolved_metadata = metadata or {}
        self._execution_store.start_run(
            run_id=resolved_run_id,
            thread_id=resolved_thread_id,
            input_payload=state,
            metadata=resolved_metadata,
        )
        token = _ACTIVE_RUN_CONTEXT.set(
            _RunContext(run_id=resolved_run_id, thread_id=resolved_thread_id, metadata=resolved_metadata)
        )
        last_node_id = None
        try:
            result = self._graph.invoke(state, config=config)
            snapshot = self._graph.get_state(config)
            if snapshot and snapshot.next:
                last_node_id = snapshot.next[0]
            elif state.get("resume_from"):
                last_node_id = state["resume_from"]
            else:
                last_node_id = "analyze"
            self._execution_store.finish_run(
                run_id=resolved_run_id,
                status="succeeded",
                output_payload=result,
                error_stack=None,
                last_node_id=last_node_id,
            )
        except Exception:
            self._execution_store.finish_run(
                run_id=resolved_run_id,
                status="failed",
                output_payload=None,
                error_stack=traceback.format_exc(),
                last_node_id=last_node_id,
            )
            raise
        finally:
            _ACTIVE_RUN_CONTEXT.reset(token)
            self._execution_store.flush()
        return {
            "run_id": resolved_run_id,
            "thread_id": resolved_thread_id,
            "question": result["question"],
            "llm_profile": result["llm_profile"],
            "rag_error": result["rag_error"],
            "analysis": result["analysis"],
            "rag_content": result["rag_content"],
            "citations": result["citations"],
        }

    def resume_from_node(
        self,
        *,
        thread_id: str,
        node_id: str,
        state_override: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target = node_id.strip().lower()
        if target not in {"retrieve", "analyze"}:
            raise ValueError(f"Unsupported resume node: {node_id}")
        config = self._build_config(thread_id)
        checkpoint_state = self._graph.get_state(config)
        base_state = dict(checkpoint_state.values) if checkpoint_state and checkpoint_state.values else {}
        if state_override:
            base_state.update(state_override)
        if not base_state.get("question"):
            raise ValueError("Cannot resume graph without question in checkpoint state")
        return self.run(
            question=base_state["question"],
            top_k=int(base_state.get("top_k", 5)),
            collection=base_state.get("collection"),
            thread_id=thread_id,
            resume=True,
            resume_from_node=target,
            state_override=base_state,
            metadata=metadata,
        )

    def get_execution_history(self, thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return self._execution_store.list_runs(thread_id=thread_id, limit=limit)

    def get_run_trace(self, run_id: str, checkpoint_limit: int = 100) -> dict[str, Any]:
        run = self._execution_store.get_run(run_id)
        if not run:
            raise ValueError(f"run_id not found: {run_id}")
        config = self._build_config(run["thread_id"])
        history = []
        for snapshot in self._graph.get_state_history(config, limit=max(checkpoint_limit, 1)):
            history.append(
                {
                    "config": snapshot.config,
                    "metadata": snapshot.metadata,
                    "values": snapshot.values,
                    "next": snapshot.next,
                    "created_at": snapshot.created_at,
                }
            )
        return {
            "run": run,
            "node_events": self._execution_store.list_events(run_id),
            "checkpoint_history": history,
        }

    def replay_state(self, run_id: str, until_event_id: str | None = None) -> dict[str, Any]:
        run = self._execution_store.get_run(run_id)
        if not run:
            raise ValueError(f"run_id not found: {run_id}")
        events = self._execution_store.list_events(run_id)
        replayed = dict(run["input_json"])
        for event in events:
            if event["phase"] == "succeeded" and event.get("state_json"):
                replayed = dict(event["state_json"])
            if until_event_id and event["event_id"] == until_event_id:
                break
        return replayed

    def close(self) -> None:
        try:
            self._execution_store.close()
        finally:
            self._checkpoint_conn.close()
