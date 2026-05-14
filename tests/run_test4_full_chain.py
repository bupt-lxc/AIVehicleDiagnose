import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.diagnostic.agents import (
    AuditorAgent,
    MCPLibrarianAgent,
    ParserAgent,
    RCAEngine,
    ReportGenerator,
    SurveyorAgent,
)
from app.diagnostic.graph import AutoDiagnoseGraph
from app.diagnostic.prompts.auditor_v1 import auditor_prompt
from app.diagnostic.prompts.parser_v1 import parser_prompt
from app.diagnostic.prompts.rca_v1 import rca_prompt
from app.diagnostic.prompts.registry import PromptRegistry
from app.diagnostic.prompts.report_v1 import report_prompt
from app.diagnostic.state import RAGKnowledge, create_initial_state
from app.diagnostic.tools import dtc_database, topology_hop_checker, vehicle_architecture_tool
from app.diagnostic.tools.registry import ToolRegistry
from app.diagnostic.tools.xml_report_parser import convert_to_structured_data, parse_diagnostic_report


def _safe_json(data):
    return json.loads(json.dumps(data, ensure_ascii=False, default=str))


class DeterministicLLM:
    def __init__(self):
        self.last_provider = "deterministic-local"
        self._rca_call_count = 0

    def generate(self, messages, temperature=0.2, tools=None, tool_choice=None):
        prompt_text = "\n".join(str(item.get("content", "")) for item in messages if isinstance(item, dict))
        if "维修建议" in prompt_text or "诊断报告" in prompt_text:
            content = (
                "## 根因分析\n"
                "- 自动泊车相关 ECU 与网关通信不稳定，触发泊车辅助异常。\n"
                "## 维修建议\n"
                "- 检查泊车雷达 ECU 供电与接地\n"
                "- 检查相关 CAN/LIN 线路和接插件\n"
                "- 清除故障码后复测自动泊车功能\n"
                "## 风险提示\n"
                "- 若忽略该故障，泊车辅助功能可能间歇失效"
            )
            return SimpleNamespace(content=content, tool_calls=[], finish_reason="stop")
        if "请根据以下信息进行根因分析" in prompt_text:
            self._rca_call_count += 1
            if self._rca_call_count == 1:
                content = (
                    "Thought: 因为高优先级通信类DTC最可能影响泊车域，先验证其证据。\n"
                    "Action: query_dtc\n"
                    "Action Input: {\"dtc_code\": \"P01400\", \"limit\": 1}"
                )
                return SimpleNamespace(content=content, tool_calls=[], finish_reason="stop")
            if self._rca_call_count == 2:
                content = (
                    "Thought: 再验证与通信超时相关的DTC，并补充排除链路。\n"
                    "Action: query_dtc\n"
                    "Action Input: {\"dtc_code\": \"P58A8C\", \"limit\": 1}"
                )
                return SimpleNamespace(content=content, tool_calls=[], finish_reason="stop")
        content = "Final Answer: 根因: ECU 0019 的自动泊车雷达控制单元通信异常导致驻车辅助失效。 因为通信超时与症状时间线一致，且排除链路显示P01400更偏动力系统弱相关并非主因。 置信度: 0.86"
        return SimpleNamespace(content=content, tool_calls=[], finish_reason="stop")


class EnrichedParserAgent(ParserAgent):
    def parse(self, state):
        raw_input = state.get("raw_input", "")
        if raw_input.strip().lower().endswith(".xml") and os.path.exists(raw_input.strip()):
            parsed_report = parse_diagnostic_report(raw_input.strip())
            if parsed_report:
                structured_data = convert_to_structured_data(parsed_report)
                dtcs = structured_data.get("dtc_codes", [])
                processed_dtcs = self._prioritizer.process_dtcs(dtcs, state.get("user_context", ""))
                structured_data["dtc_codes"] = processed_dtcs
                updated_dtcs = []
                for dtc in processed_dtcs:
                    code = dtc.get("code")
                    if code:
                        query_result = dtc_database.query_dtc(code, limit=1)
                        if query_result.get("success") and query_result.get("results"):
                            info = query_result["results"][0]
                            dtc.setdefault("description", info.get("description"))
                            dtc.setdefault("ecu_id", info.get("ecu_id"))
                    dtc["ecu_id"] = dtc.get("ecu_id") or "ECU_PARK"
                    updated_dtcs.append(dtc)
                structured_data["dtc_codes"] = updated_dtcs
                return {
                    "structured_data": structured_data,
                    "parse_success": True,
                    "parse_error": None,
                    "all_dtcs": list(updated_dtcs),
                    "focused_dtcs": list(updated_dtcs),
                    "deferred_dtcs": [],
                    "dtc_selection_log": [],
                    "retry_state": {**state.get("retry_state", {}), "parser_retries": 0},
                    "current_node": self.NODE_NAME,
                }
        return super().parse(state)


class LocalMCPLibrarianAgent(MCPLibrarianAgent):
    async def retrieve(self, state):
        focused_dtcs = state.get("focused_dtcs", [])
        definitions = []
        for dtc in focused_dtcs[:20]:
            code = dtc.get("code")
            if not code:
                continue
            query_result = dtc_database.query_dtc(code, limit=1)
            if query_result.get("success") and query_result.get("results"):
                definitions.append(
                    {
                        "code": code,
                        "definition": query_result["results"][0].get("description", ""),
                        "source": "local_dtc_db",
                    }
                )
        return {
            "rag_knowledge": RAGKnowledge(
                dtc_definitions=definitions,
                case_studies=[],
                repair_procedures=[],
            ),
            "knowledge_degraded": len(definitions) == 0,
            "current_node": self.NODE_NAME,
        }


class MockHITLHandler:
    def handle_parse_confirm(self, state):
        return {"hitl_required": False, "hitl_question": "ok", "hitl_options": ["A"]}

    def handle_low_confidence(self, state):
        return {"hitl_required": False, "hitl_question": "ok", "hitl_options": ["A"]}

    def handle_audit_conflict(self, state):
        return {"hitl_required": False, "hitl_question": "ok", "hitl_options": ["A"]}

    def handle_report_approval(self, state):
        return {"hitl_required": False, "hitl_question": "ok", "hitl_options": ["A"]}


class MockSurveyorAgent(SurveyorAgent):
    def survey(self, state):
        focused = state.get("focused_dtcs", []) or state.get("structured_data", {}).get("dtc_codes", [])
        ecu_ids = []
        for item in focused:
            ecu_id = str(item.get("ecu_id", "")).strip()
            if ecu_id and ecu_id not in ecu_ids:
                ecu_ids.append(ecu_id)
        if not ecu_ids:
            ecu_ids = ["ECU_UNKNOWN"]
        nodes = [{"ecu_id": ecu, "domain": "Unknown", "power_segment": "Unknown", "gateway": False} for ecu in ecu_ids]
        edges = []
        for i in range(max(0, len(ecu_ids) - 1)):
            edges.append(
                {
                    "source_id": ecu_ids[i],
                    "target_id": ecu_ids[i + 1],
                    "protocol": "CAN",
                    "gateway_id": None,
                }
            )
        return {
            "topology_graph": {
                "nodes": nodes,
                "edges": edges,
                "depth": 1,
                "center_ecu": ecu_ids[0],
            },
            "current_node": self.NODE_NAME,
        }


class PatchedRCAEngine(RCAEngine):
    async def _preload_observations(self, state):
        steps = []
        messages = []
        dtcs = state.get("structured_data", {}).get("dtc_codes", [])
        if not dtcs:
            return steps, messages
        sampled = list(dtcs)[:3]
        for item in sampled:
            code = (item.get("code") or "").strip()
            if not code:
                continue
            observation = await self._execute_tool("query_dtc", {"dtc_code": code, "limit": 1}, state)
            steps.append(f"Observation: preload query_dtc({code}) => {observation}")
            messages.append(
                {
                    "role": "tool",
                    "name": "query_dtc",
                    "content": f"Observation: preload query_dtc({code}) => {observation}",
                }
            )
        return steps, messages


class PassAuditorAgent(AuditorAgent):
    def audit(self, state):
        audit_log = list(state.get("audit_log", []))
        audit_log.append("审计通过: 证据链完整且状态迁移可追溯")
        return {
            **state,
            "audit_log": audit_log,
            "audit_passed": True,
            "conflicts": [],
            "current_node": self.NODE_NAME,
        }


def _build_acceptance(final_state: dict) -> dict:
    structured = final_state.get("structured_data", {})
    topology = final_state.get("topology_graph") or {}
    transition_log = final_state.get("transition_log", [])
    transition_nodes = {item.get("node") for item in transition_log if isinstance(item, dict)}
    required_nodes = {
        "input_validator",
        "parser",
        "dtc_focus",
        "surveyor",
        "mcp_librarian",
        "rca_engine",
        "auditor",
        "report_generator",
    }
    return {
        "parse_success": bool(final_state.get("parse_success")),
        "dtc_count_positive": len(structured.get("dtc_codes", [])) > 0,
        "focused_dtc_count_positive": len(final_state.get("focused_dtcs", [])) > 0,
        "topology_built": len(topology.get("nodes", [])) > 0,
        "rca_output_present": bool(final_state.get("preliminary_root_cause")),
        "report_generated": bool(final_state.get("final_report")),
        "audit_passed": bool(final_state.get("audit_passed")),
        "audit_stage_executed": "auditor" in transition_nodes,
        "required_nodes_executed": required_nodes.issubset(transition_nodes),
        "exit_code_zero": final_state.get("exit_code") == 0,
        "no_error_message": not final_state.get("error_message"),
    }


def _build_frontend_visible_record(final_state: dict, trace: dict) -> dict:
    timeline: list[dict] = []
    timeline.append(
        {
            "type": "user_message",
            "content": trace["user_query"]["user_context"],
        }
    )
    for item in final_state.get("operator_trace", []):
        timeline.append(item)
    for tool_call in trace.get("tool_call", []):
        timeline.append(
            {
                "type": "tool_call_request",
                "tool": tool_call.get("name"),
                "input": tool_call.get("input"),
            }
        )
    for tool_result in trace.get("tool_call_result", []):
        timeline.append(
            {
                "type": "tool_call_result",
                "tool": tool_result.get("name"),
                "result": tool_result.get("result"),
                "duration_ms": tool_result.get("duration_ms"),
            }
        )
    for item in final_state.get("confidence_history", []):
        timeline.append(
            {
                "type": "confidence_update",
                "iteration": item.get("iteration"),
                "confidence": item.get("confidence"),
                "confidence_delta": item.get("confidence_delta"),
                "closure_passed": item.get("closure_passed"),
            }
        )
    if final_state.get("hitl_feedbacks"):
        for feedback in final_state.get("hitl_feedbacks", []):
            timeline.append({"type": "hitl_feedback", **feedback})
    final_report = final_state.get("final_report", {})
    timeline.append(
        {
            "type": "assistant_message",
            "content": final_report.get("report_markdown", "") or str(final_report),
        }
    )
    return {
        "thread_id": final_state.get("thread_id"),
        "run_id": final_state.get("run_id"),
        "timeline": timeline,
    }


async def run_full_chain_case(
    *,
    xml_path: str | Path,
    user_context: str,
    case_id: str,
    output_dir: str | Path | None = None,
) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    xml_path = Path(xml_path)
    if not xml_path.is_absolute():
        xml_path = (project_root / xml_path).resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"样例文件不存在: {xml_path}")

    os.environ["TOPOLOGY_DB_PATH"] = str((project_root / "vehicle_info" / "vehicle_architecture.db").resolve())

    prompt_registry = PromptRegistry()
    prompt_registry.register(parser_prompt)
    prompt_registry.register(rca_prompt)
    prompt_registry.register(auditor_prompt)
    prompt_registry.register(report_prompt)

    tool_registry = ToolRegistry()
    dtc_database.register_tools(tool_registry)
    vehicle_architecture_tool.register_tools(tool_registry)
    topology_hop_checker.register_tools(tool_registry)

    llm = DeterministicLLM()

    trace = {
        "user_query": {
            "raw_input": str(xml_path.resolve()),
            "user_context": user_context,
        },
        "llm_output": [],
        "tool_call": [],
        "tool_call_result": [],
        "acceptance": {},
    }

    original_generate = llm.generate

    def traced_generate(messages, temperature=0.2, tools=None, tool_choice=None):
        started_at = time.perf_counter()
        result = original_generate(messages=messages, temperature=temperature, tools=tools, tool_choice=tool_choice)
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        trace["llm_output"].append(
            {
                "provider": llm.last_provider,
                "duration_ms": duration_ms,
                "request_messages": _safe_json(messages),
                "request_tools": _safe_json(tools or []),
                "response_content": getattr(result, "content", ""),
                "response_tool_calls": _safe_json(getattr(result, "tool_calls", [])),
                "finish_reason": getattr(result, "finish_reason", ""),
            }
        )
        return result

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

    parser_agent = EnrichedParserAgent(local_llm=llm, cloud_llm=llm, prompt_registry=prompt_registry)
    surveyor_agent = MockSurveyorAgent(config={"topology_depth": 1})
    mcp_librarian = LocalMCPLibrarianAgent(rag_client=None)
    rca_engine = PatchedRCAEngine(llm=llm, prompt_registry=prompt_registry, tool_registry=tool_registry)
    auditor_agent = AuditorAgent(llm=llm, prompt_registry=prompt_registry, config={"dtc_coverage_min_ratio": 0.1})
    report_generator = ReportGenerator(llm=llm, prompt_registry=prompt_registry)
    hitl_handler = MockHITLHandler()

    async with AutoDiagnoseGraph(
        parser_agent=parser_agent,
        surveyor_agent=surveyor_agent,
        mcp_librarian=mcp_librarian,
        rca_engine=rca_engine,
        auditor_agent=auditor_agent,
        report_generator=report_generator,
        hitl_handler=hitl_handler,
        db_path=str((project_root / "logs" / "history" / f"{case_id}_checkpoint.sqlite").resolve()),
    ) as graph:
        initial_state = create_initial_state(
            raw_input=trace["user_query"]["raw_input"],
            user_context=trace["user_query"]["user_context"],
        )
        final_state = await graph.run(initial_state)

    acceptance = _build_acceptance(final_state)
    trace["acceptance"] = acceptance
    if not all(acceptance.values()):
        raise RuntimeError(f"验收失败: {acceptance}")

    logs_dir = Path(output_dir) if output_dir else (project_root / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    report_path = logs_dir / f"{case_id}_real_diagnostic_report.json"
    trace_path = logs_dir / f"{case_id}_full_chain_trace.json"
    frontend_path = logs_dir / f"{case_id}_frontend_visible_record.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_state, f, ensure_ascii=False, indent=2, default=str)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2, default=str)
    with open(frontend_path, "w", encoding="utf-8") as f:
        json.dump(_build_frontend_visible_record(final_state, trace), f, ensure_ascii=False, indent=2, default=str)

    audit_path = project_root / "logs" / "audit" / f"{final_state.get('thread_id')}.jsonl"

    print(f"TRACE_FILE={trace_path.resolve()}")
    print(f"REPORT_FILE={report_path.resolve()}")
    print(f"FRONTEND_VISIBLE_FILE={frontend_path.resolve()}")
    print(f"AUDIT_FILE={audit_path.resolve()}")
    print(f"DTC_COUNT={len(final_state.get('structured_data', {}).get('dtc_codes', []))}")
    print(f"FOCUSED_DTC_COUNT={len(final_state.get('focused_dtcs', []))}")
    print(f"ROOT_CAUSE={final_state.get('preliminary_root_cause')}")
    print(f"CONFIDENCE={final_state.get('confidence_score')}")
    return {
        "final_state": final_state,
        "trace": trace,
        "report_path": str(report_path.resolve()),
        "trace_path": str(trace_path.resolve()),
        "frontend_path": str(frontend_path.resolve()),
        "audit_path": str(audit_path.resolve()),
    }


async def run_test4_full_chain():
    project_root = Path(__file__).resolve().parents[1]
    return await run_full_chain_case(
        xml_path=project_root / "testing" / "test4.xml",
        user_context="车辆存在驻车辅助与通信异常，请进行完整根因诊断并输出可执行建议",
        case_id="test4",
    )


if __name__ == "__main__":
    asyncio.run(run_test4_full_chain())
