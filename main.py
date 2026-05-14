from __future__ import annotations

import argparse
import json
import logging
import shlex
from typing import Any

from app.config import load_settings
from app.conversation_agent import ConversationAgent
from app.rag_client import RAGClient


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def print_reply(result: dict[str, Any]) -> None:
    print("=" * 70)
    print(f"Trace ID: {result['trace_id']}")
    print(f"Provider: {result['provider']}")
    print(f"Tool Used: {result['tool_used']}")
    print(f"Prompt Tokens: {result['prompt_tokens']}")
    print(f"Completion Tokens: {result['completion_tokens']}")
    print("-" * 70)
    print(result["content"])
    if result["tool_used"]:
        print("-" * 70)
        print("Tool Result:")
        print(result["tool_content"])
    print("=" * 70)


def print_diagnostic_result(result: dict[str, Any]) -> None:
    print("=" * 70)
    print(f"Run ID: {result['run_id']}")
    print(f"Thread ID: {result['thread_id']}")
    print(f"LLM Profile: {result['llm_profile']}")
    print(f"RAG Error: {result['rag_error']}")
    print("-" * 70)
    print(result["analysis"])
    print("-" * 70)
    print("Retrieved Context:")
    print(result["rag_content"])
    print("-" * 70)
    print("Citations:")
    print(json.dumps(result["citations"], ensure_ascii=False, indent=2))
    print("=" * 70)


def _parse_diag_options(parts: list[str]) -> tuple[str, int, str | None, str | None]:
    if len(parts) < 2:
        raise ValueError("Usage: /diag <question> [--top-k N] [--collection NAME] [--thread-id ID]")
    question_tokens: list[str] = []
    top_k = 5
    collection: str | None = None
    thread_id: str | None = None
    index = 1
    while index < len(parts):
        token = parts[index]
        if token == "--top-k":
            index += 1
            if index >= len(parts):
                raise ValueError("Missing value for --top-k")
            top_k = int(parts[index])
        elif token == "--collection":
            index += 1
            if index >= len(parts):
                raise ValueError("Missing value for --collection")
            collection = parts[index]
        elif token == "--thread-id":
            index += 1
            if index >= len(parts):
                raise ValueError("Missing value for --thread-id")
            thread_id = parts[index]
        else:
            question_tokens.append(token)
        index += 1
    question = " ".join(question_tokens).strip()
    if not question:
        raise ValueError("Question is required for /diag")
    return question, top_k, collection, thread_id


class _UnavailableGraph:
    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def reason(self) -> str:
        return self._reason

    def _raise(self) -> None:
        raise RuntimeError(self._reason)

    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._raise()
        return {}

    def resume_from_node(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._raise()
        return {}

    def get_execution_history(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self._raise()
        return []

    def get_run_trace(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._raise()
        return {}

    def replay_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._raise()
        return {}

    def close(self) -> None:
        return


def chat_loop(agent: ConversationAgent, graph: Any) -> None:
    print("Interactive mode started.")
    print(
        "Commands: /switch <provider> [model], /profile, /tools, /state, "
        "/diag, /diag_resume, /diag_history, /diag_trace, /diag_replay, /conv_export, /conv_delete, /exit",
    )
    while True:
        try:
            raw = input(">>> ").strip()
            if not raw:
                continue
            if raw == "/exit":
                print("Bye.")
                break
            if raw == "/profile":
                print(f"Active provider: {agent.active_provider_name}")
                continue
            if raw == "/tools":
                print(json.dumps(agent.describe_tools(), ensure_ascii=False, indent=2))
                continue
            if raw == "/state":
                state_json = json.dumps(agent.state.to_dict(), ensure_ascii=False, indent=2)
                print(state_json)
                continue
            parts = shlex.split(raw)
            command = parts[0] if parts else ""
            if command == "/switch":
                if len(parts) < 2:
                    print("Usage: /switch <provider> [model]")
                    continue
                provider_name = parts[1]
                model = parts[2] if len(parts) > 2 else None
                active = agent.switch_provider(provider_name=provider_name, model=model)
                print(f"Switched to provider={active}")
                continue
            if command == "/diag":
                question, top_k, collection, thread_id = _parse_diag_options(parts)
                result = graph.run(
                    question=question,
                    top_k=top_k,
                    collection=collection,
                    thread_id=thread_id,
                    metadata={"source": "chat_loop"},
                )
                print_diagnostic_result(result)
                continue
            if command == "/diag_resume":
                if len(parts) < 3:
                    print("Usage: /diag_resume <thread_id> <node_id>")
                    continue
                result = graph.resume_from_node(thread_id=parts[1], node_id=parts[2], metadata={"source": "chat_loop"})
                print_diagnostic_result(result)
                continue
            if command == "/diag_history":
                if len(parts) < 2:
                    print("Usage: /diag_history <thread_id> [limit]")
                    continue
                limit = int(parts[2]) if len(parts) > 2 else 20
                history = graph.get_execution_history(thread_id=parts[1], limit=limit)
                print(json.dumps(history, ensure_ascii=False, indent=2))
                continue
            if command == "/diag_trace":
                if len(parts) < 2:
                    print("Usage: /diag_trace <run_id> [checkpoint_limit]")
                    continue
                checkpoint_limit = int(parts[2]) if len(parts) > 2 else 100
                trace = graph.get_run_trace(parts[1], checkpoint_limit=checkpoint_limit)
                print(json.dumps(trace, ensure_ascii=False, indent=2))
                continue
            if command == "/diag_replay":
                if len(parts) < 2:
                    print("Usage: /diag_replay <run_id> [event_id]")
                    continue
                event_id = parts[2] if len(parts) > 2 else None
                replayed = graph.replay_state(parts[1], until_event_id=event_id)
                print(json.dumps(replayed, ensure_ascii=False, indent=2))
                continue
            if command == "/conv_export":
                print(json.dumps(agent.export_conversation(), ensure_ascii=False, indent=2))
                continue
            if command == "/conv_delete":
                agent.delete_conversation()
                print("Conversation deleted.")
                break
            reply = agent.ask_sync(raw)
            print_reply(
                {
                    "trace_id": reply.trace_id,
                    "provider": agent.active_provider_name,
                    "tool_used": reply.tool_used,
                    "prompt_tokens": reply.prompt_tokens,
                    "completion_tokens": reply.completion_tokens,
                    "content": reply.content,
                    "tool_content": reply.tool_payload.get("content", ""),
                },
            )
        except KeyboardInterrupt:
            print("\nBye.")
            break
        except Exception as exc:
            logging.getLogger("main").exception("Chat loop failed")
            print(f"Error: {exc}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Diagnostic Agent CLI")
    subparsers = parser.add_subparsers(dest="command")

    ask_parser = subparsers.add_parser("ask", help="Run one-shot conversation turn")
    ask_parser.add_argument("question", type=str)

    subparsers.add_parser("chat", help="Run interactive terminal mode")
    subparsers.add_parser("state", help="Print persisted state JSON")
    subparsers.add_parser("conv-export", help="Export current conversation data")
    subparsers.add_parser("conv-delete", help="Delete current conversation data")
    diag_parser = subparsers.add_parser("diag", help="Run DiagnosticGraph once")
    diag_parser.add_argument("question", type=str)
    diag_parser.add_argument("--top-k", type=int, default=5)
    diag_parser.add_argument("--collection", type=str, default="")
    diag_parser.add_argument("--thread-id", type=str, default="")

    diag_resume_parser = subparsers.add_parser("diag-resume", help="Resume DiagnosticGraph from node")
    diag_resume_parser.add_argument("thread_id", type=str)
    diag_resume_parser.add_argument("node_id", type=str)

    diag_history_parser = subparsers.add_parser("diag-history", help="List graph execution history by thread")
    diag_history_parser.add_argument("thread_id", type=str)
    diag_history_parser.add_argument("--limit", type=int, default=20)

    diag_trace_parser = subparsers.add_parser("diag-trace", help="Get graph run trace")
    diag_trace_parser.add_argument("run_id", type=str)
    diag_trace_parser.add_argument("--checkpoint-limit", type=int, default=100)

    diag_replay_parser = subparsers.add_parser("diag-replay", help="Replay graph state")
    diag_replay_parser.add_argument("run_id", type=str)
    diag_replay_parser.add_argument("--event-id", type=str, default="")

    return parser


def main() -> None:
    setup_logging()
    parser = create_parser()
    args = parser.parse_args()
    settings = load_settings()
    agent = ConversationAgent(settings)
    try:
        from app.llm_manager import LLMManager
        from app.graph_pipeline import DiagnosticGraph

        graph: Any = DiagnosticGraph(
            llm_manager=LLMManager(settings),
            rag_client=RAGClient(settings),
            checkpoint_db_path=settings.graph_checkpoint_db,
            audit_flush_every=settings.graph_audit_flush_every,
            payload_limit_bytes=settings.graph_payload_limit_bytes,
            node_history_limit=settings.graph_node_history_limit,
        )
    except Exception as exc:
        graph = _UnavailableGraph(f"DiagnosticGraph unavailable: {exc}")

    try:
        if args.command in {"diag", "diag-resume", "diag-history", "diag-trace", "diag-replay"} and isinstance(graph, _UnavailableGraph):
            raise SystemExit(graph.reason)
        if args.command == "ask":
            reply = agent.ask_sync(args.question)
            print_reply(
                {
                    "trace_id": reply.trace_id,
                    "provider": agent.active_provider_name,
                    "tool_used": reply.tool_used,
                    "prompt_tokens": reply.prompt_tokens,
                    "completion_tokens": reply.completion_tokens,
                    "content": reply.content,
                    "tool_content": reply.tool_payload.get("content", ""),
                },
            )
            return
        if args.command == "state":
            print(json.dumps(agent.state.to_dict(), ensure_ascii=False, indent=2))
            return
        if args.command == "conv-export":
            print(json.dumps(agent.export_conversation(), ensure_ascii=False, indent=2))
            return
        if args.command == "conv-delete":
            agent.delete_conversation()
            print("Conversation deleted.")
            return
        if args.command == "diag":
            result = graph.run(
                question=args.question,
                top_k=args.top_k,
                collection=args.collection.strip() or None,
                thread_id=args.thread_id.strip() or None,
                metadata={"source": "subcommand"},
            )
            print_diagnostic_result(result)
            return
        if args.command == "diag-resume":
            result = graph.resume_from_node(
                thread_id=args.thread_id,
                node_id=args.node_id,
                metadata={"source": "subcommand"},
            )
            print_diagnostic_result(result)
            return
        if args.command == "diag-history":
            history = graph.get_execution_history(thread_id=args.thread_id, limit=args.limit)
            print(json.dumps(history, ensure_ascii=False, indent=2))
            return
        if args.command == "diag-trace":
            trace = graph.get_run_trace(args.run_id, checkpoint_limit=args.checkpoint_limit)
            print(json.dumps(trace, ensure_ascii=False, indent=2))
            return
        if args.command == "diag-replay":
            replayed = graph.replay_state(args.run_id, until_event_id=args.event_id.strip() or None)
            print(json.dumps(replayed, ensure_ascii=False, indent=2))
            return
        chat_loop(agent, graph)
    except Exception as exc:
        logging.getLogger("main").exception("Program failed")
        raise SystemExit(f"Fatal error: {exc}") from exc
    finally:
        graph.close()
        agent.close()


if __name__ == "__main__":
    main()
