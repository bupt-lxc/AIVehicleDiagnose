"""Audit Logger for Diagnostic System

Responsible for structured logging of all intermediate steps, including:
- User inputs
- LLM inputs/outputs
- Tool calls/results
- State transitions (diffs)
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from deepdiff import DeepDiff

logger = logging.getLogger(__name__)

class AuditLogger:
    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = None
        self.last_state = None

    def start_session(self, session_id: str | None = None):
        self.session_id = session_id or uuid.uuid4().hex
        self.last_state = {}
        logger.info(f"[AuditLogger] Session started: {self.session_id}")

    def _json_serializer(self, obj):
        """Custom JSON serializer for non-serializable objects"""
        if isinstance(obj, (datetime, Path)):
            return str(obj)
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "json"):
            return obj.json()
        if isinstance(obj, type):
            return obj.__name__
        try:
            return str(obj)
        except:
            return "<unserializable>"

    def log_event(self, event_type: str, payload: Dict[str, Any], state: Dict[str, Any] | None = None):
        """Log a structured event with state diff"""
        if not self.session_id:
            self.start_session()

        timestamp = datetime.now(timezone.utc).isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "event_type": event_type,
            "payload": payload,
        }

        if state:
            # Clean state for diffing (remove complex objects if necessary or rely on serializer)
            # We use a serialized version of state for diffing to avoid type issues
            current_state_serialized = json.loads(json.dumps(state, default=self._json_serializer))
            
            # Calculate state diff
            if self.last_state is None:
                self.last_state = {}
                
            diff = DeepDiff(self.last_state, current_state_serialized, ignore_order=True).to_dict()
            log_entry["state_diff"] = diff
            self.last_state = current_state_serialized

        # Write to file
        log_file = self.log_dir / f"{self.session_id}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, default=self._json_serializer, ensure_ascii=False) + "\n")

    def log_llm_interaction(self, prompt: str | list, response: str, model: str):
        self.log_event("llm_interaction", {
            "model": model,
            "prompt": prompt,
            "response": response
        })

    def log_tool_call(self, tool_name: str, input_args: dict, result: Any, duration_ms: float):
        self.log_event("tool_call", {
            "tool_name": tool_name,
            "input": input_args,
            "result": result,
            "duration_ms": duration_ms
        })

    def log_node_execution(self, node_name: str, status: str, error: str | None = None):
        self.log_event("node_execution", {
            "node_name": node_name,
            "status": status,
            "error": error
        })
