from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:
        return None


@dataclass(slots=True)
class LLMProfile:
    name: str
    base_url: str
    api_key: str
    model: str
    
    def __post_init__(self):
        return None


@dataclass(slots=True)
class Settings:
    rag_server_url: str
    local_profile: LLMProfile
    online_profile: LLMProfile
    default_profile: str
    user_id: str
    session_id: str
    max_context_tokens: int
    compression_strategy: str
    compression_plugins: list[str] = field(default_factory=list)
    llm_provider_plugins: list[str] = field(default_factory=list)
    tool_plugins: list[str] = field(default_factory=list)
    mcp_retry_count: int = 2
    mcp_retry_backoff_ms: int = 300
    observability_dir: str = "./logs/observability"
    history_dir: str = "./logs/history"
    graph_checkpoint_db: str = "./logs/history/langgraph_checkpoint.sqlite"
    graph_audit_flush_every: int = 8
    graph_payload_limit_bytes: int = 65536
    graph_node_history_limit: int = 50000

    @property
    def profiles(self) -> dict[str, LLMProfile]:
        return {
            "local": self.local_profile,
            "online": self.online_profile,
        }

    def ensure_runtime_dirs(self) -> dict[str, Path]:
        base_dir = Path.cwd()
        observability = base_dir / self.observability_dir
        history = base_dir / self.history_dir
        observability.mkdir(parents=True, exist_ok=True)
        history.mkdir(parents=True, exist_ok=True)
        return {"observability": observability, "history": history}


def _parse_plugins(raw_value: str) -> list[str]:
    items = [item.strip() for item in raw_value.split(",")]
    return [item for item in items if item]


def _read_positive_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _read_online_api_key() -> str:
    online_api_key = os.getenv("ONLINE_LLM_API_KEY", "").strip()
    if online_api_key:
        return online_api_key
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if deepseek_api_key:
        return deepseek_api_key
    return os.getenv("DASHSCOPE_API_KEY", "").strip()


def _resolve_default_profile(online_api_key: str) -> str:
    profile = os.getenv("DEFAULT_LLM_PROFILE", "").strip().lower()
    if profile in {"local", "online"}:
        return profile
    return "online" if online_api_key else "local"


def load_settings() -> Settings:
    load_dotenv()
    online_api_key = _read_online_api_key()
    local_profile = LLMProfile(
        name="local",
        base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("LOCAL_LLM_API_KEY", "not-used"),
        model=os.getenv("LOCAL_LLM_MODEL", "Qwen3.5-4B"),
    )
    online_profile = LLMProfile(
        name="online",
        base_url=os.getenv("ONLINE_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=online_api_key,
        model=os.getenv("ONLINE_LLM_MODEL", "qvq-max-2025-05-15"),
    )
    return Settings(
        rag_server_url=os.getenv("RAG_SERVER_URL", "http://localhost:7300"),
        local_profile=local_profile,
        online_profile=online_profile,
        default_profile=_resolve_default_profile(online_api_key),
        user_id=os.getenv("USER_ID", "default-user"),
        session_id=os.getenv("SESSION_ID", "default-session"),
        max_context_tokens=_read_positive_int("MAX_CONTEXT_TOKENS", 128000),
        compression_strategy=os.getenv("COMPRESSION_STRATEGY", "summary").strip() or "summary",
        compression_plugins=_parse_plugins(os.getenv("COMPRESSION_PLUGINS", "")),
        llm_provider_plugins=_parse_plugins(os.getenv("LLM_PROVIDER_PLUGINS", "")),
        tool_plugins=_parse_plugins(os.getenv("TOOL_PLUGINS", "")),
        mcp_retry_count=_read_positive_int("MCP_RETRY_COUNT", 2),
        mcp_retry_backoff_ms=_read_positive_int("MCP_RETRY_BACKOFF_MS", 300),
        observability_dir=os.getenv("OBSERVABILITY_DIR", "logs/observability"),
        history_dir=os.getenv("HISTORY_DIR", "logs/history"),
        graph_checkpoint_db=os.getenv("GRAPH_CHECKPOINT_DB", "logs/history/langgraph_checkpoint.sqlite"),
        graph_audit_flush_every=_read_positive_int("GRAPH_AUDIT_FLUSH_EVERY", 8),
        graph_payload_limit_bytes=_read_positive_int("GRAPH_PAYLOAD_LIMIT_BYTES", 65536),
        graph_node_history_limit=_read_positive_int("GRAPH_NODE_HISTORY_LIMIT", 50000),
    )
