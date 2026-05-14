from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MetricsCounter:
    token_input: int = 0
    token_output: int = 0
    compression_count: int = 0
    mcp_call_count: int = 0
    mcp_error_count: int = 0
    mcp_total_latency_ms: float = 0.0
    rca_iteration_count: int = 0
    confidence_delta_total: float = 0.0
    confidence_delta_count: int = 0
    focused_dtc_size_total: int = 0
    focused_dtc_size_count: int = 0
    deferred_dtc_recall_count: int = 0
    hitl_trigger_count: int = 0
    transition_error_count: int = 0
    transition_total_count: int = 0
    restart_recovery_success: int = 0

    def as_dict(self) -> dict[str, Any]:
        error_rate = (self.mcp_error_count / self.mcp_call_count) if self.mcp_call_count else 0.0
        avg_latency = (self.mcp_total_latency_ms / self.mcp_call_count) if self.mcp_call_count else 0.0
        avg_conf_delta = (self.confidence_delta_total / self.confidence_delta_count) if self.confidence_delta_count else 0.0
        avg_focused_size = (self.focused_dtc_size_total / self.focused_dtc_size_count) if self.focused_dtc_size_count else 0.0
        transition_error_rate = (self.transition_error_count / self.transition_total_count) if self.transition_total_count else 0.0
        return {
            "token_input": self.token_input,
            "token_output": self.token_output,
            "compression_count": self.compression_count,
            "mcp_call_count": self.mcp_call_count,
            "mcp_error_count": self.mcp_error_count,
            "mcp_error_rate": round(error_rate, 6),
            "mcp_avg_latency_ms": round(avg_latency, 3),
            "rca_iteration_count": self.rca_iteration_count,
            "average_confidence_delta": round(avg_conf_delta, 6),
            "focused_dtc_size": round(avg_focused_size, 3),
            "deferred_dtc_recall_count": self.deferred_dtc_recall_count,
            "hitl_trigger_rate": round((self.hitl_trigger_count / self.transition_total_count) if self.transition_total_count else 0.0, 6),
            "transition_error_rate": round(transition_error_rate, 6),
            "restart_recovery_success": self.restart_recovery_success,
        }


@dataclass(slots=True)
class AuditContext:
    user_id: str
    session_id: str
    trace_id: str
    turn_id: int
    metadata: dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self._output_dir / f"audit-{time.strftime('%Y%m%d')}.jsonl"
        self._metrics_path = self._output_dir / "metrics.json"
        self._lock = threading.Lock()
        self._metrics = MetricsCounter()

    @property
    def file_path(self) -> Path:
        return self._file_path

    @property
    def metrics_path(self) -> Path:
        return self._metrics_path

    def update_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self._metrics.token_input += max(prompt_tokens, 0)
            self._metrics.token_output += max(completion_tokens, 0)
            self._flush_metrics_locked()

    def record_compression(self) -> None:
        with self._lock:
            self._metrics.compression_count += 1
            self._flush_metrics_locked()

    def record_mcp_call(self, latency_ms: float, is_error: bool) -> None:
        with self._lock:
            self._metrics.mcp_call_count += 1
            self._metrics.mcp_total_latency_ms += max(latency_ms, 0.0)
            if is_error:
                self._metrics.mcp_error_count += 1
            self._flush_metrics_locked()

    def emit(self, event_type: str, context: AuditContext, payload: dict[str, Any]) -> None:
        body: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "event_type": event_type,
            "user_id": context.user_id,
            "session_id": context.session_id,
            "trace_id": context.trace_id,
            "turn_id": context.turn_id,
            "metadata": context.metadata,
            "payload": payload,
        }
        line = json.dumps(body, ensure_ascii=False)
        with self._lock:
            with self._file_path.open("a", encoding="utf-8") as f:
                f.write(f"{line}\n")
            self._apply_event_metrics_locked(event_type, payload)

    def _apply_event_metrics_locked(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "rca_iteration":
            self._metrics.rca_iteration_count += 1
            delta = float(payload.get("confidence_delta", 0.0) or 0.0)
            self._metrics.confidence_delta_total += delta
            self._metrics.confidence_delta_count += 1
        elif event_type == "dtc_focus":
            size = int(payload.get("focused_dtc_size", 0) or 0)
            self._metrics.focused_dtc_size_total += max(size, 0)
            self._metrics.focused_dtc_size_count += 1
        elif event_type == "deferred_dtc_recall":
            self._metrics.deferred_dtc_recall_count += int(payload.get("count", 0) or 0)
        elif event_type == "hitl_triggered":
            self._metrics.hitl_trigger_count += 1
        elif event_type == "state_transition":
            self._metrics.transition_total_count += 1
            if payload.get("result") == "error":
                self._metrics.transition_error_count += 1
        elif event_type == "restart_recovery":
            if payload.get("success"):
                self._metrics.restart_recovery_success += 1
        self._flush_metrics_locked()

    def _flush_metrics_locked(self) -> None:
        self._metrics_path.write_text(
            json.dumps(self._metrics.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
