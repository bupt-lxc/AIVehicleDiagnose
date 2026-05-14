import asyncio
import logging
import os
import time
from typing import Dict, Any

from langgraph.graph import StateGraph, END, START
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except Exception:
    try:
        from langgraph_checkpoint.sqlite.aio import AsyncSqliteSaver
    except Exception:
        AsyncSqliteSaver = None

from app.diagnostic.state import DiagnosticState
from app.diagnostic.validators import InputValidator
from app.diagnostic.agents import (
    ParserAgent,
    SurveyorAgent,
    MCPLibrarianAgent,
    RCAEngine,
    AuditorAgent,
    ReportGenerator
)
from app.diagnostic.hitl.handler import HITLHandler
from app.config import Settings, load_settings
from app.rag_client import RAGClient
from app.providers import OpenAICompatibleProvider
from app.diagnostic.prompts.registry import PromptRegistry
from app.diagnostic.tools.registry import ToolRegistry
from app.diagnostic.audit_logger import AuditLogger
from app.diagnostic.prioritizer import DTCFocusEngine

from app.diagnostic.prompts.parser_v1 import parser_prompt
from app.diagnostic.prompts.rca_v1 import rca_prompt
from app.diagnostic.prompts.auditor_v1 import auditor_prompt
from app.diagnostic.prompts.report_v1 import report_prompt

logger = logging.getLogger(__name__)

class CheckpointManager:
    _instance = None
    _checkpointer_cm = None
    _checkpointer = None
    _lock = None

    @classmethod
    async def get_checkpointer(cls, db_path: str):
        import asyncio
        if cls._lock is None:
            cls._lock = asyncio.Lock()
            
        async with cls._lock:
            if db_path == ":memory:":
                raise RuntimeError("Checkpoint必须使用SQLite持久化文件，禁止使用内存模式(:memory:).")
            if AsyncSqliteSaver is None:
                raise RuntimeError("未检测到SQLite Checkpoint依赖，请安装 langgraph-checkpoint-sqlite。")
            if cls._checkpointer is None:
                import os
                if db_path != ":memory:":
                    os.makedirs(os.path.dirname(db_path), exist_ok=True)
                cls._checkpointer_cm = AsyncSqliteSaver.from_conn_string(db_path)
                cls._checkpointer = await cls._checkpointer_cm.__aenter__()
            return cls._checkpointer

class AutoDiagnoseGraph:
    MAX_PARSER_RETRIES = 3
    MAX_RCA_RETRIES = 3
    MAX_AUDIT_RETRIES = 2
    MAX_REVISIONS = 10  # 增加最大修改次数
    MIN_CONFIDENCE = 0.7
    
    def __init__(
        self,
        parser_agent: ParserAgent = None,
        surveyor_agent: SurveyorAgent = None,
        mcp_librarian: MCPLibrarianAgent = None,
        rca_engine: RCAEngine = None,
        auditor_agent: AuditorAgent = None,
        report_generator: ReportGenerator = None,
        hitl_handler: HITLHandler = None,
        db_path: str = "logs/diagnostic/checkpoint.sqlite"
    ):
        settings = load_settings()
        rag_client = RAGClient(settings=settings)
        
        # Initialize default providers if agents are not passed
        preferred_profile = settings.online_profile if settings.online_profile.api_key else settings.local_profile
        default_llm = OpenAICompatibleProvider(profile=preferred_profile)
        
        prompt_registry = PromptRegistry()
        # Register default prompts
        prompt_registry.register(parser_prompt)
        prompt_registry.register(rca_prompt)
        prompt_registry.register(auditor_prompt)
        prompt_registry.register(report_prompt)
        
        tool_registry = ToolRegistry()
        tool_registry.load_plugins([
            "app/diagnostic/tools/dtc_database.py",
            "app/diagnostic/tools/dtc_query_tool.py",
            "app/diagnostic/tools/dtc_parser.py",
            "app/diagnostic/tools/freeze_frame_parser.py",
            "app/diagnostic/tools/vehicle_architecture_tool.py",
            "app/diagnostic/tools/topology_hop_checker.py",
        ])
        
        self.parser_agent = parser_agent or ParserAgent(
            cloud_llm=default_llm, 
            prompt_registry=prompt_registry
        )
        self.surveyor_agent = surveyor_agent or SurveyorAgent()
        self.mcp_librarian = mcp_librarian or MCPLibrarianAgent(rag_client=rag_client)
        self.rca_engine = rca_engine or RCAEngine(
            llm=default_llm,
            prompt_registry=prompt_registry,
            tool_registry=tool_registry
        )
        self.auditor_agent = auditor_agent or AuditorAgent(
            llm=default_llm,
            prompt_registry=prompt_registry
        )
        # Note: AuditorAgent accepts prompt_registry (used for future LLM-based audit expansion)
        self.report_generator = report_generator or ReportGenerator(
            llm=default_llm,
            prompt_registry=prompt_registry
        )
        self.hitl_handler = hitl_handler or HITLHandler()
        self.dtc_focus_engine = DTCFocusEngine(max_focused=12)
        self.db_path = db_path
        self.audit_logger = AuditLogger()
        
        self.workflow = self._build_workflow()
        self.graph = None
        self.checkpointer = None
        self._checkpointer_cm = None

    def _build_workflow(self):
        graph = StateGraph(DiagnosticState)
        
        # 添加节点
        graph.add_node("input_validator", self._input_validator_node)
        graph.add_node("parser", self._parser_node)
        graph.add_node("dtc_focus", self._dtc_focus_node)
        graph.add_node("parser_fallback", self._parser_fallback_node)
        graph.add_node("hitl_parse_confirm", self._hitl_parse_confirm_node)
        graph.add_node("surveyor", self._surveyor_node)
        graph.add_node("mcp_librarian", self._mcp_librarian_node)
        graph.add_node("rca_engine", self._rca_engine_node)
        graph.add_node("hitl_low_confidence", self._hitl_low_confidence_node)
        graph.add_node("auditor", self._auditor_node)
        graph.add_node("hitl_audit_conflict", self._hitl_audit_conflict_node)
        graph.add_node("hitl_report_approval", self._hitl_report_approval_node)
        graph.add_node("report_generator", self._report_generator_node)
        graph.add_node("error_handler", self._error_handler_node)
        
        # 添加边
        graph.add_edge(START, "input_validator")
        
        graph.add_conditional_edges(
            "input_validator",
            self._route_after_validation,
            {"parser": "parser", "end": END}
        )
        
        graph.add_conditional_edges(
            "parser",
            self._route_after_parser,
            {
                "focus": "dtc_focus",
                "retry": "parser",
                "fallback": "parser_fallback",
                "error": "error_handler",
            }
        )
        graph.add_conditional_edges(
            "dtc_focus",
            self._route_after_focus,
            {"hitl": "hitl_parse_confirm", "surveyor": "surveyor"}
        )
        
        graph.add_conditional_edges(
            "parser_fallback",
            self._route_after_fallback,
            {"hitl": "hitl_parse_confirm", "error": "error_handler"}
        )
        
        graph.add_edge("hitl_parse_confirm", "surveyor")
        graph.add_edge("surveyor", "mcp_librarian")
        graph.add_edge("mcp_librarian", "rca_engine")
        
        graph.add_conditional_edges(
            "rca_engine",
            self._route_after_rca,
            {"hitl": "hitl_low_confidence", "auditor": "auditor", "retry": "rca_engine", "error": "error_handler"}
        )
        
        graph.add_conditional_edges(
            "hitl_low_confidence",
            self._route_after_low_confidence,
            {"auditor": "auditor", "retry": "rca_engine"}
        )
        
        graph.add_conditional_edges(
            "auditor",
            self._route_after_audit,
            {"hitl": "hitl_audit_conflict", "approval": "hitl_report_approval"}
        )
        
        graph.add_conditional_edges(
            "hitl_audit_conflict",
            self._route_after_audit_conflict,
            {"retry": "rca_engine", "approval": "hitl_report_approval"}
        )
        
        graph.add_conditional_edges(
            "hitl_report_approval",
            self._route_after_approval,
            {
                "approved": "report_generator",
                "revise": "rca_engine",
                "more_info": "hitl_low_confidence",
            }
        )
        
        graph.add_edge("report_generator", END)
        graph.add_edge("error_handler", END)
        
        return graph

    async def __aenter__(self):
        self.checkpointer = await CheckpointManager.get_checkpointer(self.db_path)
        if self.checkpointer is None:
            raise RuntimeError("SQLite Checkpoint初始化失败，无法进入可回溯模式。")
        interrupt_nodes = [
            "hitl_parse_confirm",
            "hitl_low_confidence",
            "hitl_audit_conflict",
            "hitl_report_approval",
        ]
        self.graph = self.workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_after=interrupt_nodes
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # We don't close the checkpointer here since it's shared
        pass

    async def run(self, state: DiagnosticState, config: dict = None, interactive: bool = False,
              on_waiting=None, on_node_event=None, on_hitl_display=None, on_hitl_input=None) -> DiagnosticState:
        if not self.graph:
            raise RuntimeError("Graph not compiled. Use 'async with AutoDiagnoseGraph() ...'")
        
        self.audit_logger.start_session(state.get("thread_id"))
        self.audit_logger.log_event("graph_start", {"initial_state": state})
        
        try:
            run_config = config or {"configurable": {"thread_id": state["thread_id"]}}
            if not state.get("degraded_mode") and not load_settings().online_profile.api_key:
                state["degraded_mode"] = True
                state["risk_flags"] = list(state.get("risk_flags", [])) + ["offline_degraded_mode"]

            hitl_nodes = [
                "hitl_parse_confirm",
                "hitl_low_confidence",
                "hitl_audit_conflict",
                "hitl_report_approval"
            ]

            async def _astream_with_polling(input_state):
                last_poll = time.time()

                async def _stream():
                    async for chunk in self.graph.astream(input_state, run_config, stream_mode="updates"):
                        for node_name, node_output in chunk.items():
                            if on_node_event:
                                await on_node_event(node_name, node_output)

                task = asyncio.create_task(_stream())
                while not task.done():
                    await asyncio.sleep(2)
                    if on_waiting:
                        await on_waiting()
                await task

                snapshot = await self.graph.aget_state(run_config)
                return snapshot.values if snapshot.values else {}

            final_state = await _astream_with_polling(state)
            resume_count = 0
            MAX_RESUME_RETRIES = 10
            HITL_INTERACTION_LIMIT = 20
            hitl_interaction_count = 0

            while True:
                if resume_count >= MAX_RESUME_RETRIES:
                    logger.error(f"[Graph] Auto-resume limit exceeded ({MAX_RESUME_RETRIES}). Stopping loop.")
                    final_state["error_message"] = "Auto-resume limit exceeded"
                    final_state["exit_code"] = 9
                    break

                current_node = final_state.get("current_node")

                if current_node in hitl_nodes:
                    if final_state.get("hitl_required") is False:
                        logger.info(f"[Graph] Auto-resuming from {current_node} as HITL not required")
                        self.audit_logger.log_event("auto_resume", {"node": current_node})
                        await self.graph.aupdate_state(run_config, {"hitl_required": False})
                        final_state = await _astream_with_polling(None)
                        resume_count += 1
                    elif interactive:
                        hitl_interaction_count += 1
                        if hitl_interaction_count > HITL_INTERACTION_LIMIT:
                            logger.error(f"[Graph] HITL interaction limit ({HITL_INTERACTION_LIMIT}) exceeded. Breaking loop.")
                            final_state["error_message"] = "HITL interaction limit exceeded"
                            final_state["exit_code"] = 9
                            break

                        hitl_question = final_state.get("hitl_question", "No question provided")
                        hitl_options = final_state.get("hitl_options", [])

                        if on_hitl_display:
                            await on_hitl_display(current_node, hitl_question, hitl_options)

                        if on_hitl_input:
                            user_choice, user_input = await on_hitl_input(current_node, hitl_question, hitl_options)
                        else:
                            print(f"\n{'='*60}")
                            print(f"[HITL] 当前节点: {current_node}")
                            print(f"[HITL] 问题: {hitl_question}")
                            print(f"[HITL] 选项: {', '.join(str(o) for o in hitl_options)}")
                            print(f"{'='*60}")
                            user_choice = input("\n请输入您的选择 (A/B/C/D/Z 等): ").strip().upper()
                            if not user_choice:
                                user_choice = "A"
                            user_input = None

                        processed = self.hitl_handler.process_feedback(final_state, user_choice, user_input)
                        await self.graph.aupdate_state(run_config, {
                            "hitl_feedbacks": processed.get("hitl_feedbacks", []),
                            "hitl_required": processed.get("hitl_required", False),
                            "current_hitl_point": processed.get("current_hitl_point"),
                            "conflict_ignored": processed.get("conflict_ignored", False),
                            "human_hint": processed.get("human_hint", ""),
                            "user_description": processed.get("user_description", ""),
                            "user_context": processed.get("user_context", ""),
                            "risk_flags": processed.get("risk_flags", []),
                            "report_approved": processed.get("report_approved", False),
                            "priority_dtcs": processed.get("priority_dtcs", []),
                        })

                        final_state = await _astream_with_polling(None)
                    else:
                        logger.warning(f"[Graph] Stopped at HITL node {current_node} (interactive=False)")
                        break
                else:
                    break

            self.audit_logger.log_event("graph_end", {"final_state": final_state}, state=final_state)
            return final_state
        except Exception as e:
            self.audit_logger.log_event("graph_error", {"error": str(e)})
            raise e

    # ==============================
    # Nodes
    # ==============================

    def _build_transition_entry(
        self,
        *,
        state: DiagnosticState,
        node_name: str,
        started_at: float,
        output: dict[str, Any],
        result: str,
    ) -> dict[str, Any]:
        return {
            "node": node_name,
            "timestamp": time.time(),
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "result": result,
            "input_summary": {
                "focused_dtc_count": len(state.get("focused_dtcs", [])),
                "deferred_dtc_count": len(state.get("deferred_dtcs", [])),
                "rca_iteration": int(state.get("rca_iteration", 0) or 0),
                "hitl_required": bool(state.get("hitl_required", False)),
            },
            "output_summary": {
                "current_node": output.get("current_node"),
                "error_message": output.get("error_message"),
                "exit_code": output.get("exit_code", state.get("exit_code")),
            },
        }

    def _append_transition(self, state: DiagnosticState, node_name: str, started_at: float, output: dict[str, Any], result: str) -> dict[str, Any]:
        entry = self._build_transition_entry(
            state=state,
            node_name=node_name,
            started_at=started_at,
            output=output,
            result=result,
        )
        transition_log = list(output.get("transition_log", state.get("transition_log", [])))
        transition_log.append(entry)
        operator_trace = list(output.get("operator_trace", state.get("operator_trace", [])))
        operator_trace.append(
            {
                "type": "transition",
                "node": node_name,
                "result": result,
                "latency_ms": entry["latency_ms"],
                "summary": entry["output_summary"],
            }
        )
        output["transition_log"] = transition_log
        output["operator_trace"] = operator_trace
        self.audit_logger.log_event(
            "state_transition",
            {
                "node": node_name,
                "result": result,
                "latency_ms": entry["latency_ms"],
            },
        )
        return output

    def _input_validator_node(self, state: DiagnosticState) -> Dict[str, Any]:
        started_at = time.perf_counter()
        self.audit_logger.log_event("node_start", {"node": "input_validator"}, state=state)
        validator = InputValidator()
        result = validator.validate_raw_input(state.get("raw_input", ""))
        
        if result.is_valid:
            output = {"current_node": "input_validator", "exit_code": 0}
        else:
            output = {
                "current_node": "input_validator",
                "exit_code": 1,
                "error_message": "; ".join(result.errors)
            }
        status = "success" if result.is_valid else "failed"
        self.audit_logger.log_node_execution("input_validator", status, output.get("error_message"))
        return self._append_transition(state, "input_validator", started_at, output, status)

    def _parser_node(self, state: DiagnosticState) -> Dict[str, Any]:
        started_at = time.perf_counter()
        self.audit_logger.log_event("node_start", {"node": "parser"}, state=state)
        
        import copy
        state_copy = copy.deepcopy(state)
        result = self.parser_agent.parse(state_copy)
        
        # Ensure all nested structures are preserved
        merged_result = {**result}
        if "structured_data" in result:
            merged_result["structured_data"] = {
                **state_copy.get("structured_data", {}),
                **result.get("structured_data", {})
            }
        if "retry_state" in result:
            merged_result["retry_state"] = {
                **state_copy.get("retry_state", {}),
                **result.get("retry_state", {})
            }
            
        status = "success" if result.get("parse_success") else "failed"
        self.audit_logger.log_node_execution("parser", status, result.get("parse_error"))
        return self._append_transition(state, "parser", started_at, merged_result, status)

    def _dtc_focus_node(self, state: DiagnosticState) -> Dict[str, Any]:
        started_at = time.perf_counter()
        self.audit_logger.log_event("node_start", {"node": "dtc_focus"}, state=state)
        structured = dict(state.get("structured_data", {}))
        all_dtcs = list(state.get("all_dtcs") or structured.get("dtc_codes", []))
        preferred_ecus: list[str] = []
        hitl_feedbacks = state.get("hitl_feedbacks", [])
        if hitl_feedbacks:
            last = hitl_feedbacks[-1]
            user_input = str(last.get("user_input") or "")
            preferred_ecus = [item.strip().upper() for item in user_input.replace(",", " ").split() if len(item.strip()) >= 2]
        focused_result = self.dtc_focus_engine.focus(
            dtcs=all_dtcs,
            user_context=state.get("user_context", ""),
            preferred_ecus=preferred_ecus,
        )
        structured["dtc_codes"] = list(focused_result["focused_dtcs"])
        output = {
            "current_node": "dtc_focus",
            "structured_data": structured,
            "all_dtcs": focused_result["all_dtcs"],
            "focused_dtcs": focused_result["focused_dtcs"],
            "deferred_dtcs": focused_result["deferred_dtcs"],
            "dtc_selection_log": focused_result["dtc_selection_log"],
            "hitl_required": False,
            "error_message": None,
        }
        if len(focused_result["focused_dtcs"]) == 0 and len(focused_result["all_dtcs"]) > 0:
            output["hitl_required"] = True
            output["current_hitl_point"] = "focus_adjustment"
            output["hitl_question"] = "未筛选出重点DTC，请指定优先ECU或补充故障现象。"
            output["hitl_options"] = ["A", "D"]
        self.audit_logger.log_node_execution("dtc_focus", "success")
        return self._append_transition(state, "dtc_focus", started_at, output, "success")

    def _parser_fallback_node(self, state: DiagnosticState) -> Dict[str, Any]:
        started_at = time.perf_counter()
        self.audit_logger.log_event("node_start", {"node": "parser_fallback"}, state=state)
        logger.warning("[Graph] Executing parser fallback")
        
        # CHAIN-001: 尝试从原始输入中提取至少一些信息
        raw_input = state.get("raw_input", "")
        # 这里使用简单的正则提取作为最后手段
        import re
        extracted_dtcs = re.findall(r"[PBCU]\d{4}", raw_input)
        
        if extracted_dtcs:
             dtc_list = [{"code": code, "status": "Unknown"} for code in extracted_dtcs]
             output = {
                "current_node": "parser_fallback",
                "structured_data": {"dtc_codes": dtc_list, "ecu_states": [], "freeze_frames": [], "raw_text_segments": []},
                "parse_success": False,  # 标记为非完全成功
                "parse_error": "解析失败，已通过正则提取部分DTC",
                "hitl_required": True,  # 触发人工确认
                "current_hitl_point": "parse_confirm", # 复用 parse_confirm 逻辑
                "exit_code": 2
            }
             return self._append_transition(state, "parser_fallback", started_at, output, "fallback")
        
        result = {
            "current_node": "parser_fallback",
            "exit_code": 2,  # Fallback triggered
            "error_message": "Parser fallback triggered, manual intervention or retry required."
        }
        self.audit_logger.log_node_execution("parser_fallback", "executed")
        return self._append_transition(state, "parser_fallback", started_at, result, "fallback")

    def _hitl_parse_confirm_node(self, state: DiagnosticState) -> Dict[str, Any]:
        started_at = time.perf_counter()
        self.audit_logger.log_event("node_start", {"node": "hitl_parse_confirm"}, state=state)
        result = self.hitl_handler.handle_parse_confirm(state)
        result["current_node"] = "hitl_parse_confirm"
        self.audit_logger.log_node_execution("hitl_parse_confirm", "waiting", f"Question: {result.get('hitl_question')}")
        return self._append_transition(state, "hitl_parse_confirm", started_at, result, "waiting")

    def _surveyor_node(self, state: DiagnosticState) -> Dict[str, Any]:
        started_at = time.perf_counter()
        self.audit_logger.log_event("node_start", {"node": "surveyor"}, state=state)
        result = self.surveyor_agent.survey(state)
        self.audit_logger.log_node_execution("surveyor", "success")
        return self._append_transition(state, "surveyor", started_at, result, "success")

    async def _mcp_librarian_node(self, state: DiagnosticState) -> Dict[str, Any]:
        started_at = time.perf_counter()
        self.audit_logger.log_event("node_start", {"node": "mcp_librarian"}, state=state)
        result = await self.mcp_librarian.retrieve(state)
        self.audit_logger.log_node_execution("mcp_librarian", "success")
        return self._append_transition(state, "mcp_librarian", started_at, result, "success")

    async def _rca_engine_node(self, state: DiagnosticState) -> Dict[str, Any]:
        started_at = time.perf_counter()
        self.audit_logger.log_event("node_start", {"node": "rca_engine"}, state=state)
        result = await self.rca_engine.analyze(state)
        status = "success" if not result.get("error_message") else "failed"
        self.audit_logger.log_node_execution("rca_engine", status, result.get("error_message"))
        return self._append_transition(state, "rca_engine", started_at, result, status)

    def _hitl_low_confidence_node(self, state: DiagnosticState) -> Dict[str, Any]:
        started_at = time.perf_counter()
        self.audit_logger.log_event("node_start", {"node": "hitl_low_confidence"}, state=state)
        result = self.hitl_handler.handle_low_confidence(state)
        result["current_node"] = "hitl_low_confidence"
        self.audit_logger.log_node_execution("hitl_low_confidence", "waiting")
        return self._append_transition(state, "hitl_low_confidence", started_at, result, "waiting")

    def _auditor_node(self, state: DiagnosticState) -> Dict[str, Any]:
        started_at = time.perf_counter()
        self.audit_logger.log_event("node_start", {"node": "auditor"}, state=state)
        result = self.auditor_agent.audit(state)
        status = "success" if result.get("audit_passed") else "failed"
        self.audit_logger.log_node_execution("auditor", status, f"Conflicts: {len(result.get('conflicts', []))}")
        return self._append_transition(state, "auditor", started_at, result, status)

    def _hitl_audit_conflict_node(self, state: DiagnosticState) -> Dict[str, Any]:
        started_at = time.perf_counter()
        self.audit_logger.log_event("node_start", {"node": "hitl_audit_conflict"}, state=state)
        result = self.hitl_handler.handle_audit_conflict(state)
        result["current_node"] = "hitl_audit_conflict"
        self.audit_logger.log_node_execution("hitl_audit_conflict", "waiting")
        return self._append_transition(state, "hitl_audit_conflict", started_at, result, "waiting")

    def _hitl_report_approval_node(self, state: DiagnosticState) -> Dict[str, Any]:
        started_at = time.perf_counter()
        self.audit_logger.log_event("node_start", {"node": "hitl_report_approval"}, state=state)
        result = self.hitl_handler.handle_report_approval(state)
        result["current_node"] = "hitl_report_approval"
        self.audit_logger.log_node_execution("hitl_report_approval", "waiting")
        return self._append_transition(state, "hitl_report_approval", started_at, result, "waiting")

    def _report_generator_node(self, state: DiagnosticState) -> Dict[str, Any]:
        started_at = time.perf_counter()
        self.audit_logger.log_event("node_start", {"node": "report_generator"}, state=state)
        result = self.report_generator.generate(state)
        self.audit_logger.log_node_execution("report_generator", "success")
        return self._append_transition(state, "report_generator", started_at, result, "success")

    def _error_handler_node(self, state: DiagnosticState) -> Dict[str, Any]:
        started_at = time.perf_counter()
        self.audit_logger.log_event("node_start", {"node": "error_handler"}, state=state)
        logger.error(f"[Graph] Error occurred. Node: {state.get('current_node')}, Error: {state.get('error_message')}")
        result = {"current_node": "error_handler", "exit_code": 1}
        self.audit_logger.log_node_execution("error_handler", "executed")
        return self._append_transition(state, "error_handler", started_at, result, "error")

    # ==============================
    # Edges
    # ==============================

    def _route_after_validation(self, state: DiagnosticState) -> str:
        if state.get("exit_code") == 0:
            return "parser"
        return "end"

    def _route_after_parser(self, state: DiagnosticState) -> str:
        if state.get("parse_success"):
            return "focus"
        retry_state = state.get("retry_state", {})
        retries = retry_state.get("parser_retries", 0)
        if retries < self.MAX_PARSER_RETRIES:
            return "retry"
        else:
            return "fallback"

    def _route_after_focus(self, state: DiagnosticState) -> str:
        if state.get("hitl_required"):
            return "hitl"
        return "surveyor"

    def _route_after_fallback(self, state: DiagnosticState) -> str:
        # CHAIN-001: 即使 parse_success 为 False，如果有 DTC 也可以继续
        dtc_codes = state.get("structured_data", {}).get("dtc_codes")
        if state.get("parse_success", False) or dtc_codes:
            return "hitl"
        return "error"

    def _route_after_rca(self, state: DiagnosticState) -> str:
        if state.get("hitl_required"):
            return "hitl"

        # Check if there is an error message from RCA
        error_msg = state.get("error_message")
        if error_msg and state.get("current_node") == "rca_engine":
            retry_state = state.get("retry_state", {})
            is_recoverable = bool(state.get("error_recoverable", True))
            if is_recoverable and retry_state.get("rca_retries", 0) < self.MAX_RCA_RETRIES:
                return "retry"
            if not is_recoverable:
                return "error"
            return "auditor"
                
        confidence = state.get("confidence_score")
        
        # STATE-003: None 或 0.0 表示未计算，直接进入审计（可能是RCA失败或其他情况）
        if confidence is None or confidence == 0.0:
            return "auditor"
            
        # LOGIC-003: 只有当置信度有效且低于阈值时才触发HITL
        if 0 < confidence <= self.MIN_CONFIDENCE:
            return "hitl"
        return "auditor"

    def _route_after_low_confidence(self, state: DiagnosticState) -> str:
        hitl_feedbacks = state.get("hitl_feedbacks", [])
        last_low_choice = None
        for fb in reversed(hitl_feedbacks):
            if fb.get("point") == "hitl_low_confidence":
                last_low_choice = fb.get("user_choice", "")
                break
        if last_low_choice:
            possible_causes = self.hitl_handler._extract_possible_causes(state.get("analysis_steps", []))
            other_letter = chr(65 + len(possible_causes))
            if last_low_choice == other_letter:
                return "retry"
        return "auditor"

    def _route_after_audit(self, state: DiagnosticState) -> str:
        if state.get("audit_passed"):
            return "approval"
        return "hitl"

    def _route_after_audit_conflict(self, state: DiagnosticState) -> str:
        hitl_feedbacks = state.get("hitl_feedbacks", [])
        last_audit_choice = None
        for fb in reversed(hitl_feedbacks):
            if fb.get("point") == "hitl_audit_conflict":
                last_audit_choice = fb.get("user_choice", "").upper()
                break
        if last_audit_choice:
            if last_audit_choice.startswith("B") or last_audit_choice.startswith("C"):
                return "retry"
        return "approval"

    def _route_after_approval(self, state: DiagnosticState) -> str:
        hitl_feedbacks = state.get("hitl_feedbacks", [])
        retry_state = state.get("retry_state", {})
        revision_count = retry_state.get("revision_count", 0)
        
        last_approval_choice = None
        for fb in reversed(hitl_feedbacks):
            if fb.get("point") == "hitl_report_approval":
                last_approval_choice = str(fb.get("user_choice", "")).upper()
                break
                
        if last_approval_choice:
            if last_approval_choice.startswith("A"):
                return "approved"
            elif last_approval_choice.startswith("B"):
                if revision_count >= self.MAX_REVISIONS:
                    logger.warning("达到最大修改次数，强制批准")
                    return "approved"
                return "revise"
            elif last_approval_choice.startswith("C"):
                return "more_info"
        return "approved"
