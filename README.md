# AI Disgnostic

## Overview

This project is upgraded into an auditable CLI Agent with:

- Persistent multi-turn conversation loop
- Configurable context limit and pluggable compression
- MCP-compatible tool calling with trace-id and retry
- Hot-swappable LLM provider interface
- Structured JSON observability logs and metrics

## Architecture

```text
User CLI
  |
  v
ConversationAgent
  |- ContextManager (max tokens + compression strategies)
  |- ProviderRegistry (LLMProvider hot swap)
  |- ToolRegistry (dynamic tools + MCP query tool)
  |- RAGClient (MCP SSE protocol, trace-id, retry)
  |- ConversationStore (state + history persistence)
  |- AuditLogger (JSON audit + metrics backend)
```

## Key Paths

- `main.py`: CLI loop and command entry
- `app/conversation_agent.py`: end-to-end orchestration
- `app/session.py`: context control and persistence
- `app/compression.py`: pluggable compression strategies
- `app/providers.py`: `LLMProvider` interface and registry
- `app/rag_client.py`: MCP protocol client with trace and retry
- `app/observability.py`: structured JSON logs and metrics
- `tests/`: unit tests

## CLI Commands

- `python main.py chat`: start interactive loop
- `python main.py ask "your question"`: single-turn call
- `python main.py state`: print persisted conversation state

If local LLM endpoint is unavailable, agent returns a fallback response and records `llm_call_error` in audit logs.

Interactive commands:

- `/switch <provider> [model]`
- `/profile`
- `/tools`
- `/state`
- `/exit`

## Configuration

Copy `.env.example` to `.env`, then adjust values.

Important keys:

- `DEFAULT_LLM_PROFILE`: default `online` for DashScope/Bailian
- `MAX_CONTEXT_TOKENS`: default `30000`
- `COMPRESSION_STRATEGY`: `summary` / `sliding_window` / `vectorized`
- `COMPRESSION_PLUGINS`: comma-separated plugin classes (`module:ClassName`)
- `LLM_PROVIDER_PLUGINS`: comma-separated provider classes (`module:ClassName`)
- `TOOL_PLUGINS`: comma-separated callables (`module:function_name`)
- `ONLINE_LLM_BASE_URL`, `ONLINE_LLM_API_KEY`, `ONLINE_LLM_MODEL`
- `ONLINE_LLM_FALLBACK_MODELS`: model fallback list for online provider
- `DASHSCOPE_API_KEY`: fallback key for online profile
- `MCP_RETRY_COUNT`, `MCP_RETRY_BACKOFF_MS`
- `OBSERVABILITY_DIR`, `HISTORY_DIR`

## Compression Policy

When context exceeds the configured token limit:

1. Keep the latest three rounds (six messages)
2. Merge older messages with the previous compressed memory
3. Re-compress using the selected pluggable strategy
4. Persist compression events into audit logs

## MCP Compatibility

The agent calls existing MCP tools through SSE endpoint:

- Default endpoint: `${RAG_SERVER_URL}/sse`
- Tool call payload includes `trace_id`
- Automatic retries on failure
- Each tool call is written into audit backend

## Observability

Structured JSON audit log:

- Path: `logs/observability/audit-YYYYMMDD.jsonl`
- Includes user/session/trace/turn/event payload

Metrics file:

- Path: `logs/observability/metrics.json`
- Token usage
- Compression count
- MCP average latency
- MCP error rate

## Tests

Run:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Coverage scope:

- Context compression workflow
- LLM provider hot-plug loading
- MCP call retry and trace-id propagation

## One-click (Windows)

Double-click:

- `run_one_click.bat`

It installs dependencies, runs unit tests, then starts CLI chat.

## Test Report

- Report file: `tests/test_report.json`
- Includes test command, pass/fail status, and summary
