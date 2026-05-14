import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_settings
from app.diagnostic.agents import SurveyorAgent, AuditorAgent
from app.diagnostic.graph import AutoDiagnoseGraph
from app.diagnostic.prompts.auditor_v1 import auditor_prompt
from app.diagnostic.prompts.parser_v1 import parser_prompt
from app.diagnostic.prompts.rca_v1 import rca_prompt
from app.diagnostic.prompts.registry import PromptRegistry
from app.diagnostic.prompts.report_v1 import report_prompt
from app.diagnostic.state import create_initial_state
from app.diagnostic.tools import dtc_database
from app.diagnostic.tools import vehicle_architecture_tool
from app.diagnostic.tools.registry import ToolRegistry
from app.providers import OpenAICompatibleProvider
from tests.test_real_flow import (
    EnrichedParserAgent,
    LocalMCPLibrarianAgent,
    MockHITLHandler,
    PatchedRCAEngine,
    PatchedReportGenerator,
    patched_get_sub_topology,
)


def _safe_json(data):
    return json.loads(json.dumps(data, ensure_ascii=False, default=str))


class ResilientLLM:
    def __init__(self, online_provider):
        self.online_provider = online_provider
        self.last_provider = "online"

    def generate(self, messages, temperature=0.2, tools=None, tool_choice=None):
        result = self.online_provider.generate(
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
        )
        self.last_provider = "online"
        return result


async def run_test2_full_chain():
    settings = load_settings()
    settings.default_profile = "online"
    os.environ["ONLINE_LLM_FALLBACK_MODELS"] = ",".join(
        [
            "qwen3-max-2025-09-23",
            "qwen-plus-2025-12-01",
            "qwen-turbo-latest",
            "qwen2.5-32b-instruct",
            "qwen-plus-2025-04-28",
            "qwen-plus-2025-01-25",
        ]
    )
    _PROJECT_ROOT = Path(__file__).parent.parent
    os.environ["TOPOLOGY_DB_PATH"] = str(_PROJECT_ROOT / "vehicle_info" / "vehicle_architecture.db")
    import app.diagnostic.agents.surveyor_agent as surveyor_module
    surveyor_module.get_sub_topology = patched_get_sub_topology

    prompt_registry = PromptRegistry()
    prompt_registry.register(parser_prompt)
    prompt_registry.register(rca_prompt)
    prompt_registry.register(auditor_prompt)
    prompt_registry.register(report_prompt)

    tool_registry = ToolRegistry()
    dtc_database.register_tools(tool_registry)
    vehicle_architecture_tool.register_tools(tool_registry)

    online_llm = OpenAICompatibleProvider(profile=settings.online_profile, timeout_seconds=120)
    llm = ResilientLLM(online_llm)

    trace = {
        "user_query": {
            "raw_input": str(_PROJECT_ROOT / "testing" / "test2.xml"),
            "user_context": "该车自动泊车存在异常，请分析故障根因并给出诊断报告",
        },
        "llm_output": [],
        "tool_call": [],
        "tool_call_result": [],
    }

    original_generate = llm.generate

    def traced_generate(messages, temperature=0.2, tools=None, tool_choice=None):
        started_at = time.perf_counter()
        try:
            result = original_generate(
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
            )
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            trace["llm_output"].append(
                {
                    "provider": llm.last_provider,
                    "model": settings.online_profile.model,
                    "duration_ms": duration_ms,
                    "request_messages": _safe_json(messages),
                    "request_tools": _safe_json(tools or []),
                    "response_content": result.content,
                    "response_tool_calls": _safe_json(result.tool_calls),
                    "finish_reason": result.finish_reason,
                }
            )
            return result
        except Exception as e:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            trace["llm_output"].append(
                {
                    "provider": llm.last_provider,
                    "model": settings.online_profile.model,
                    "duration_ms": duration_ms,
                    "request_messages": _safe_json(messages),
                    "request_tools": _safe_json(tools or []),
                    "error": str(e),
                }
            )
            raise

    llm.generate = traced_generate

    for tool_name, tool_def in tool_registry._tools.items():
        original_handler = tool_def.handler

        def traced_handler_factory(name, handler):
            def traced_handler(**kwargs):
                started_at = time.perf_counter()
                trace["tool_call"].append({"name": name, "input": _safe_json(kwargs)})
                result = handler(**kwargs)
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                trace["tool_call_result"].append(
                    {
                        "name": name,
                        "result": str(result),
                        "duration_ms": duration_ms,
                    }
                )
                return result

            traced_handler.__name__ = handler.__name__
            return traced_handler

        tool_def.handler = traced_handler_factory(tool_name, original_handler)

    parser_agent = EnrichedParserAgent(cloud_llm=llm, prompt_registry=prompt_registry)
    surveyor_agent = SurveyorAgent(config={"topology_depth": 1})
    mcp_librarian = LocalMCPLibrarianAgent(rag_client=None)
    rca_engine = PatchedRCAEngine(llm=llm, prompt_registry=prompt_registry, tool_registry=tool_registry)
    auditor_agent = AuditorAgent(llm=llm, prompt_registry=prompt_registry)
    report_generator = PatchedReportGenerator(llm=llm, prompt_registry=prompt_registry)
    hitl_handler = MockHITLHandler()

    async with AutoDiagnoseGraph(
        parser_agent=parser_agent,
        surveyor_agent=surveyor_agent,
        mcp_librarian=mcp_librarian,
        rca_engine=rca_engine,
        auditor_agent=auditor_agent,
        report_generator=report_generator,
        hitl_handler=hitl_handler,
        db_path=settings.graph_checkpoint_db,
    ) as graph:
        initial_state = create_initial_state(
            raw_input=trace["user_query"]["raw_input"],
            user_context=trace["user_query"]["user_context"],
        )
        final_state = await graph.run(initial_state)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    report_path = logs_dir / "test2_real_diagnostic_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_state, f, ensure_ascii=False, indent=2, default=str)

    trace["final_report_path"] = str(report_path.resolve())
    trace["thread_id"] = final_state.get("thread_id")
    trace["parse_success"] = final_state.get("parse_success")
    trace["dtc_count"] = len(final_state.get("structured_data", {}).get("dtc_codes", []))
    trace["analysis_steps_count"] = len(final_state.get("analysis_steps", []))
    trace["root_cause"] = final_state.get("preliminary_root_cause")
    trace["confidence_score"] = final_state.get("confidence_score")
    trace["exit_code"] = final_state.get("exit_code")
    trace["error_message"] = final_state.get("error_message")

    trace_path = logs_dir / "test2_full_chain_trace.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2, default=str)

    print(f"TRACE_FILE={trace_path.resolve()}")
    print(f"REPORT_FILE={report_path.resolve()}")
    print(f"PARSE_SUCCESS={final_state.get('parse_success')}")
    print(f"DTC_COUNT={trace['dtc_count']}")
    print(f"ANALYSIS_STEPS={trace['analysis_steps_count']}")
    print(f"ROOT_CAUSE={trace['root_cause']}")
    print(f"ERROR_MESSAGE={trace['error_message']}")


if __name__ == "__main__":
    asyncio.run(run_test2_full_chain())
