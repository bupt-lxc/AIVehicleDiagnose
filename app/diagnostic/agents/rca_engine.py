"""RCA Engine - 根因推理大脑

基于 ReAct 模式的故障根因推理引擎。
结合拓扑图和 RAG 知识库进行推理分析。

依赖:
    - diagnostic.prompts.registry
    - diagnostic.tools.registry
    - diagnostic.state
    - diagnostic.validators
    - app.providers (LLM)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from app.diagnostic.prompts.registry import PromptRegistry, DynamicPromptTemplate
from app.diagnostic.tools.registry import ToolRegistry, ProgressiveDisclosureController
from app.diagnostic.state import DiagnosticState
from app.diagnostic.state_checker import StateIntegrityChecker

if TYPE_CHECKING:
    from app.providers import LLMProvider

logger = logger.bind(component="RCAEngine")

_MAX_OBSERVATION_CHARS = 600

# ── Terminal ANSI ──────────────────────────────────────────────
_G = "\033[92m"
_Y = "\033[93m"
_B = "\033[94m"
_C = "\033[96m"
_M = "\033[95m"
_R = "\033[91m"
_D = "\033[1m"
_X = "\033[0m"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _safe_lower(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lower()


@dataclass(slots=True)
class RCAConfig:
    """RCA 引擎配置"""
    min_confidence: float = 0.7
    max_retries: int = 3
    max_steps: int = 10
    timeout_seconds: float = 120.0
    enable_progressive_disclosure: bool = True
    verbose: bool = True


@dataclass(slots=True)
class ReasoningStep:
    """推理步骤"""
    thought: str
    action: str | None
    action_input: dict | None
    observation: str | None
    timestamp: str


class RCAError(Exception):
    """RCA 推理错误"""
    
    def __init__(self, message: str, step: int = 0, recoverable: bool = True):
        self.message = message
        self.step = step
        self.recoverable = recoverable
        super().__init__(message)


class RCAEngine:
    """根因推理引擎"""
    
    NODE_NAME = "rca_engine"
    
    def __init__(
        self,
        llm: "LLMProvider",
        prompt_registry: PromptRegistry,
        tool_registry: ToolRegistry,
        config: RCAConfig | None = None,
    ):
        self._llm = llm
        self._prompt_registry = prompt_registry
        self._tool_registry = tool_registry
        self._config = config or RCAConfig()
        self._state_checker = StateIntegrityChecker()
        
        if self._config.enable_progressive_disclosure:
            self._disclosure = ProgressiveDisclosureController(tool_registry)
        else:
            self._disclosure = None
        self._last_disclosure_event: dict[str, Any] | None = None
        
        logger.info(f"[RCAEngine] 初始化完成, min_confidence={self._config.min_confidence}, verbose={self._config.verbose}")
    
    def _observation_message(self, content: str) -> dict[str, str]:
        return {"role": "user", "content": content}

    def _assistant_message_from_response(self, response: Any) -> dict[str, Any]:
        content = getattr(response, "content", "") or ""
        message: dict[str, Any] = {"role": "assistant", "content": content}
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            message["tool_calls"] = tool_calls
        raw = getattr(response, "raw", None)
        if isinstance(raw, dict):
            choices = raw.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    raw_message = first_choice.get("message")
                    if isinstance(raw_message, dict):
                        reasoning_content = raw_message.get("reasoning_content")
                        if isinstance(reasoning_content, str) and reasoning_content:
                            message["reasoning_content"] = reasoning_content
        return message

    def _parse_tool_call(self, tool_call: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        func = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        action = func.get("name")
        args_str = func.get("arguments", "{}")
        try:
            action_input = json.loads(args_str)
        except Exception:
            action_input = {"query": args_str}
        if not isinstance(action_input, dict):
            action_input = {"query": action_input}
        return action, action_input

    def _emit(self, category: str, message: str, detail: Any = None) -> None:
        """统一终端日志输出（verbose模式）"""
        if not self._config.verbose:
            return
        prefix_map = {
            "init": f"{_D}[{_ts()}]{_X} {_C}[初始化]{_X} RCAEngine",
            "state": f"{_D}[{_ts()}]{_X} {_B}[状态检查]{_X}",
            "preload": f"{_D}[{_ts()}]{_X} {_M}[预加载]{_X}",
            "thought": f"{_D}[{_ts()}]{_X} {_M}[思考]{_X}",
            "action": f"{_D}[{_ts()}]{_X} {_Y}[工具调用]{_X}",
            "observation": f"{_D}[{_ts()}]{_X} {_G}[观察结果]{_X}",
            "confidence": f"{_D}[{_ts()}]{_X} {_C}[置信度]{_X}",
            "final": f"{_D}[{_ts()}]{_X} {_G}[最终结论]{_X}",
            "error": f"{_D}[{_ts()}]{_X} {_R}[错误]{_X}",
            "warn": f"{_D}[{_ts()}]{_X} {_Y}[警告]{_X}",
            "tool_disclosure": f"{_D}[{_ts()}]{_X} {_B}[工具层级]{_X}",
            "hypothesis": f"{_D}[{_ts()}]{_X} {_M}[假设]{_X}",
            "rejected": f"{_D}[{_ts()}]{_X} {_R}[排除假设]{_X}",
            "closure": f"{_D}[{_ts()}]{_X} {_Y}[闭环保检查]{_X}",
        }
        prefix = prefix_map.get(category, f"{_D}[{_ts()}]{_X}")
        line = f"{prefix} {message}"
        if detail is not None:
            detail_str = str(detail)
            if len(detail_str) > 300:
                detail_str = detail_str[:297] + "..."
            line += f" | {detail_str}"
        print(line, flush=True)
    
    async def analyze(self, state: DiagnosticState) -> DiagnosticState:
        """执行根因分析
        
        Args:
            state: 当前诊断状态
            
        Returns:
            更新后的诊断状态
        """
        structured_data = state.get("structured_data", {})
        dtc_codes = structured_data.get("dtc_codes", [])
        focused_count = len(state.get("focused_dtcs", []))
        deferred_count = len(state.get("deferred_dtcs", []))
        self._emit("init", f"开始根因分析 | DTC总数={len(dtc_codes)} 重点={focused_count} 延迟={deferred_count}")
        
        integrity_report = self._state_checker.check(state)
        
        if not integrity_report.is_valid:
            self._emit("error", f"状态完整性检查失败: {', '.join(integrity_report.missing_fields)}")
            logger.error(f"[RCAEngine] 状态完整性检查失败: {integrity_report.missing_fields}")
            retry_state = state.get("retry_state", {})
            retry_state["rca_retries"] = retry_state.get("rca_retries", 0) + 1
            return {
                **state,
                "retry_state": retry_state,
                "error_message": f"状态不完整: {', '.join(integrity_report.missing_fields)}",
                "error_recoverable": False,
                "exit_code": 9,  # System Error
                "current_node": self.NODE_NAME,
            }
        
        structured_data = state.get("structured_data", {})
        dtc_codes = structured_data.get("dtc_codes", [])
        
        if not dtc_codes:
            self._emit("warn", "无DTC数据可分析")
            logger.warning("[RCAEngine] 无 DTC 数据可分析")
            retry_state = state.get("retry_state", {})
            retry_state["rca_retries"] = retry_state.get("rca_retries", 0) + 1
            return {
                **state,
                "retry_state": retry_state,
                "error_message": "无 DTC 数据可分析",
                "error_recoverable": True,
                "current_node": self.NODE_NAME,
            }

        topology_nodes = (state.get("topology_graph") or {}).get("nodes", [])
        if not topology_nodes:
            self._emit("warn", "未获取到有效拓扑图，触发HITL")
            return {
                **state,
                "hitl_required": True,
                "current_hitl_point": "low_confidence",
                "hitl_question": "未获取到有效拓扑图，请确认是否补充目标ECU或网络症状后重试。",
                "hitl_options": ["D", "A"],
                "error_message": "拓扑图为空",
                "current_node": self.NODE_NAME,
            }

        dtc_definitions = (state.get("rag_knowledge") or {}).get("dtc_definitions", [])
        min_required = max(1, min(5, len(dtc_codes) // 4))
        if len(dtc_definitions) < min_required:
            self._emit("warn", f"DTC知识召回不足: {len(dtc_definitions)}/{min_required}，触发HITL")
            return {
                **state,
                "hitl_required": True,
                "current_hitl_point": "low_confidence",
                "hitl_question": "DTC 知识召回不足，建议补充故障现象或优先 ECU 以提高召回质量。",
                "hitl_options": ["D", "A"],
                "error_message": f"DTC 知识召回不足: {len(dtc_definitions)}/{min_required}",
                "current_node": self.NODE_NAME,
            }
        
        retry_state = state.get("retry_state", {})
        rca_retries = retry_state.get("rca_retries", 0)
        
        if rca_retries >= self._config.max_retries:
            self._emit("error", f"达到最大重试次数({self._config.max_retries})，终止分析")
            logger.error("[RCAEngine] 达到最大重试次数")
            return {
                **state,
                "error_message": "达到最大重试次数",
                "error_recoverable": False,
                "exit_code": 5,  # RCA Failure
                "current_node": self.NODE_NAME,
            }
        
        try:
            self._emit("init", f"启动ReAct循环 | 最大步数={self._config.max_steps} 超时={self._config.timeout_seconds}s")
            result = await self._run_react_loop(state)
            existing_history = list(state.get("confidence_history", []))
            existing_history.extend(result.get("confidence_history", []))
            operator_trace = list(state.get("operator_trace", []))
            operator_trace.extend(result.get("operator_trace", []))
            internal_reasoning = list(state.get("internal_reasoning", []))
            internal_reasoning.extend(result.get("internal_reasoning", []))
            
            final_confidence = result["confidence"]
            self._emit("final", f"ReAct循环完成 | 步数={result.get('iteration_count', 0)} 置信度={final_confidence:.0%} 根因={str(result['root_cause'])[:100]}")
            
            return {
                **state,
                "analysis_steps": result["steps"],
                "confidence_score": result["confidence"],
                "confidence_breakdown": result.get("confidence_breakdown", state.get("confidence_breakdown", {})),
                "preliminary_root_cause": result["root_cause"],
                "confidence_history": existing_history,
                "rca_iteration": result.get("iteration_count", 0),
                "evidence_graph": result.get("evidence_graph", state.get("evidence_graph", {"nodes": [], "edges": []})),
                "hypothesis_queue": result.get("hypothesis_queue", []),
                "hypothesis_candidates": result.get("hypothesis_candidates", []),
                "rejected_hypotheses": result.get("rejected_hypotheses", []),
                "operator_trace": operator_trace,
                "internal_reasoning": internal_reasoning,
                "error_message": None,
                "current_node": self.NODE_NAME,
            }
            
        except RCAError as e:
            self._emit("error", f"RCA推理错误(步骤{e.step}): {e.message} | 可恢复={e.recoverable}")
            logger.error(f"[RCAEngine] 推理错误 (步骤 {e.step}): {e.message}")
            
            retry_state["rca_retries"] = rca_retries + 1
            need_hitl = bool(e.recoverable)
            
            return {
                **state,
                "retry_state": retry_state,
                "error_message": e.message,
                "error_recoverable": bool(e.recoverable),
                "analysis_steps": [],
                "confidence_score": 0.0,
                "confidence_breakdown": {
                    "evidence_sufficiency": 0.0,
                    "topology_consistency": 0.0,
                    "temporal_consistency": 0.0,
                    "historical_prior_match": 0.0,
                    "aggregated_confidence": 0.0,
                },
                "preliminary_root_cause": None,
                "hitl_required": need_hitl,
                "current_hitl_point": "low_confidence" if need_hitl else state.get("current_hitl_point"),
                "hitl_question": "RCA 推理遇到歧义，是否补充现象、时间线或更关注的 ECU？" if need_hitl else state.get("hitl_question"),
                "hitl_options": ["A", "D"] if need_hitl else state.get("hitl_options", []),
                "current_node": self.NODE_NAME,
            }
        
        except Exception as e:
            self._emit("error", f"RCA未知错误: {str(e)[:200]}")
            logger.exception(f"[RCAEngine] 未知错误: {e}")
            
            retry_state["rca_retries"] = rca_retries + 1
            
            return {
                **state,
                "retry_state": retry_state,
                "error_message": f"未知错误: {str(e)}",
                "error_recoverable": False,
                "current_node": self.NODE_NAME,
            }
    
    async def _run_react_loop(self, state: DiagnosticState) -> dict:
        steps: list[str] = []
        confidence_history: list[dict[str, Any]] = []
        operator_trace: list[dict[str, Any]] = []
        internal_reasoning: list[dict[str, Any]] = []
        hypothesis_queue: list[dict[str, Any]] = []
        evidence_graph: dict[str, Any] = self._initialize_evidence_graph(state)
        deferred_recalled = False
        step_count = 0
        start_time = time.time()
        
        def _push_trace(item: dict[str, Any]) -> None:
            """推入operator_trace并实时更新state"""
            operator_trace.append(item)
            if self._config.verbose:
                existing = list(state.get("operator_trace", []))
                existing.append(item)
                state["operator_trace"] = existing
        
        system_prompt = self._prompt_registry.get_system_prompt("rca")
        
        messages = [{"role": "system", "content": system_prompt}]
        
        user_prompt = self._build_user_prompt(state)
        messages.append({"role": "user", "content": user_prompt})

        preload_steps, preload_messages = await self._preload_observations(state)
        if preload_steps:
            self._emit("preload", f"预加载{len(preload_steps)}条观察结果")
            steps.extend(preload_steps)
            messages.extend(preload_messages)
            for item in preload_steps:
                _push_trace({"type": "preload_observation", "detail": item})
        
        while step_count < self._config.max_steps:
            if time.time() - start_time > self._config.timeout_seconds:
                self._emit("error", f"推理超时({self._config.timeout_seconds}s)，步骤{step_count}")
                raise RCAError("推理超时", step=step_count, recoverable=True)
            
            available_tools = self._get_available_tools(state)
            if getattr(self, "_last_disclosure_event", None):
                _push_trace(
                    {
                        "type": "tool_disclosure_level_change",
                        **self._last_disclosure_event,
                        "iteration": step_count + 1,
                    }
                )
                self._emit("tool_disclosure", f"Level {self._last_disclosure_event.get('new_level')} | {self._last_disclosure_event.get('reason')}")
            before_conf = confidence_history[-1]["confidence"] if confidence_history else state.get("confidence_score", 0.0)
            
            response = self._llm.generate(
                messages=messages,
                tools=available_tools,
                temperature=0.1,
            )
            
            assistant_message = response.content
            messages.append(self._assistant_message_from_response(response))
            
            parsed = self._parse_react_response(assistant_message)
            
            # 兼容原生 tool calling
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_call = response.tool_calls[0]
                parsed["action"], parsed["action_input"] = self._parse_tool_call(tool_call)
                if not parsed["thought"] and not assistant_message:
                    parsed["thought"] = f"Calling tool {parsed['action']}"
            else:
                # 如果没有 tool_calls，且没有解析出 action，也没有解析出 final_answer，那么把它当做 final_answer
                if not parsed["action"] and not parsed["final_answer"]:
                    parsed["final_answer"] = assistant_message
            
            if parsed["thought"]:
                steps.append(f"Thought: {parsed['thought']}")
                hypothesis_queue.append({"iteration": step_count + 1, "thought": parsed["thought"]})
                internal_reasoning.append({"iteration": step_count + 1, "thought": parsed["thought"]})
                _push_trace(
                    {
                        "type": "react_thought",
                        "iteration": step_count + 1,
                        "content": parsed["thought"],
                    }
                )
                self._emit("thought", f"Iter{step_count+1}: {parsed['thought'][:150]}")
            
            if parsed["final_answer"]:
                root_cause = self._extract_root_cause(parsed["final_answer"])
                closure_check = self._evaluate_root_cause_closure(state, steps, root_cause)
                confidence_breakdown = self._calculate_confidence_breakdown(state, steps, root_cause, closure_check)
                confidence = min(confidence_breakdown["aggregated_confidence"], closure_check["max_confidence"])
                self._emit("closure", f"闭环保检查: {'通过' if closure_check['passed'] else '未通过'} | 备注={'; '.join(closure_check.get('notes', []))[:100]}")
                if not closure_check["passed"] and state.get("deferred_dtcs") and not deferred_recalled:
                    seed_ecu = self._pick_recall_seed_ecu(state)
                    if seed_ecu:
                        recall_obs = await self._execute_tool("recall_deferred_dtcs", {"ecu_id": seed_ecu, "limit": 8}, state)
                        steps.append(f"Observation: recall_deferred_dtcs({seed_ecu}) => {self._truncate_str(str(recall_obs), _MAX_OBSERVATION_CHARS)}")
                        _push_trace(
                            {
                                "type": "tool_call",
                                "iteration": step_count + 1,
                                "tool": "recall_deferred_dtcs",
                                "input": {"ecu_id": seed_ecu, "limit": 8},
                                "observation": str(recall_obs)[:1200],
                            }
                        )
                        self._emit("action", f"召回延迟DTC(ECU={seed_ecu}) | 结果={str(recall_obs)[:100]}")
                        messages.append(
                            self._observation_message(f"Observation: recall_deferred_dtcs({seed_ecu}) => {recall_obs}")
                        )
                        deferred_recalled = True
                        step_count += 1
                        continue
                hypothesis_candidates, rejected_hypotheses = self._build_hypothesis_outputs(
                    state=state,
                    root_cause=root_cause,
                    confidence_breakdown=confidence_breakdown,
                    closure_check=closure_check,
                )
                self._finalize_evidence_graph(
                    state=state,
                    evidence_graph=evidence_graph,
                    root_cause=root_cause,
                    hypothesis_candidates=hypothesis_candidates,
                    rejected_hypotheses=rejected_hypotheses,
                    closure_check=closure_check,
                )
                _push_trace(
                    {
                        "type": "hypothesis_candidates",
                        "items": hypothesis_candidates,
                    }
                )
                self._emit("hypothesis", f"假设候选: {len(hypothesis_candidates)}个")
                for hc in hypothesis_candidates[:3]:
                    self._emit("hypothesis", f"  #{hc.get('rank')}: {str(hc.get('hypothesis', ''))[:120]} (score={hc.get('score', 0):.4f})")
                _push_trace(
                    {
                        "type": "rejected_hypotheses",
                        "items": rejected_hypotheses,
                    }
                )
                if rejected_hypotheses:
                    self._emit("rejected", f"排除假设: {len(rejected_hypotheses)}个")
                    for rh in rejected_hypotheses[:2]:
                        self._emit("rejected", f"  {str(rh.get('hypothesis', ''))[:100]}: {str(rh.get('reason', ''))[:100]}")
                _push_trace(
                    {
                        "type": "react_final_answer",
                        "iteration": step_count + 1,
                        "content": parsed["final_answer"],
                        "root_cause": root_cause,
                        "confidence": confidence,
                        "confidence_breakdown": confidence_breakdown,
                        "closure_passed": closure_check["passed"],
                        "closure_notes": closure_check["notes"],
                    }
                )
                self._emit("final", f"Iter{step_count+1} 最终结论: 置信度={confidence:.0%} 闭环保={'通过' if closure_check['passed'] else '未通过'} | 根因={root_cause[:150]}")
                self._emit("confidence", f"置信度分解: evidence={confidence_breakdown.get('evidence_sufficiency', 0):.3f} topology={confidence_breakdown.get('topology_consistency', 0):.3f} temporal={confidence_breakdown.get('temporal_consistency', 0):.3f} historical={confidence_breakdown.get('historical_prior_match', 0):.3f} => aggregated={confidence_breakdown.get('aggregated_confidence', 0):.3f}")
                confidence_history.append(
                    {
                        "iteration": step_count + 1,
                        "confidence": confidence,
                        "confidence_delta": round(confidence - float(before_conf or 0.0), 6),
                        "tools_used": [],
                        "new_evidence_count": 0,
                        "closure_passed": closure_check["passed"],
                        "confidence_breakdown": confidence_breakdown,
                    }
                )
                
                return {
                    "steps": steps,
                    "confidence": confidence,
                    "confidence_breakdown": confidence_breakdown,
                    "root_cause": root_cause,
                    "confidence_history": confidence_history,
                    "iteration_count": step_count + 1,
                    "operator_trace": operator_trace,
                    "internal_reasoning": internal_reasoning,
                    "hypothesis_queue": hypothesis_queue,
                    "hypothesis_candidates": hypothesis_candidates,
                    "rejected_hypotheses": rejected_hypotheses,
                    "evidence_graph": evidence_graph,
                }
            
            if parsed["action"] and parsed["action_input"]:
                native_tool_calls = response.tool_calls if hasattr(response, "tool_calls") and response.tool_calls else []
                pending_actions: list[tuple[dict[str, Any] | None, str, dict[str, Any]]] = []
                if native_tool_calls:
                    for tool_call in native_tool_calls:
                        action, action_input = self._parse_tool_call(tool_call)
                        if action:
                            pending_actions.append((tool_call, action, action_input))
                else:
                    pending_actions.append((None, parsed["action"], parsed["action_input"]))

                if len(pending_actions) > 1:
                    observations: list[tuple[dict[str, Any] | None, str, dict[str, Any], str]] = []
                    for action_index, (tool_call, action, action_input) in enumerate(pending_actions, start=1):
                        steps.append(f"Action: {action}({action_input})")
                        _push_trace(
                            {
                                "type": "react_action",
                                "iteration": step_count + 1,
                                "tool": action,
                                "input": action_input,
                            }
                        )
                        self._emit("action", f"Iter{step_count+1} è°ƒç”¨å·¥å…·: {action} | è¾“å…¥={str(action_input)[:150]}")

                        observation = await self._execute_tool(action, action_input, state)
                        observation_text = str(observation)
                        observations.append((tool_call, action, action_input, observation_text))
                        steps.append(f"Observation: {self._truncate_str(observation_text, _MAX_OBSERVATION_CHARS)}")
                        _push_trace(
                            {
                                "type": "tool_call",
                                "iteration": step_count + 1,
                                "tool": action,
                                "input": action_input,
                                "observation": observation_text[:1200],
                            }
                        )
                        self._emit("observation", f"Iter{step_count+1} å·¥å…·ç»“æžœ({action}): {observation_text[:200]}")
                        evidence_graph["nodes"].append(
                            self._make_node(
                                node_id=f"obs_{step_count + 1}_{action_index}",
                                node_type="TIMELINE",
                                label=f"Observation {step_count + 1}.{action_index}",
                                attributes={
                                    "tool": action,
                                    "observation": observation_text[:400],
                                    "iteration": step_count + 1,
                                },
                            )
                        )
                        self._enrich_evidence_graph_from_observation(
                            state=state,
                            evidence_graph=evidence_graph,
                            action=action,
                            action_input=action_input,
                            observation=observation_text,
                        )

                    confidence_candidate = max(
                        [self._estimate_confidence_from_observation(observation, before_conf) for _, _, _, observation in observations],
                        default=float(before_conf or 0.0),
                    )
                    confidence_history.append(
                        {
                            "iteration": step_count + 1,
                            "confidence": confidence_candidate,
                            "confidence_delta": round(confidence_candidate - float(before_conf or 0.0), 6),
                            "tools_used": [action for _, action, _, _ in observations],
                            "new_evidence_count": len(observations),
                        }
                    )
                    self._emit("confidence", f"Iter{step_count+1} ç½®ä¿¡åº¦æ›´æ–°: {before_conf:.3f} â†’ {confidence_candidate:.3f} (Î”={confidence_candidate - float(before_conf or 0.0):+.4f})")

                    for tool_call, action, _action_input, observation in observations:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id") if tool_call else None,
                            "name": action,
                            "content": observation,
                        })
                    step_count += 1
                    continue

                steps.append(f"Action: {parsed['action']}({parsed['action_input']})")
                _push_trace(
                    {
                        "type": "react_action",
                        "iteration": step_count + 1,
                        "tool": parsed["action"],
                        "input": parsed["action_input"],
                    }
                )
                self._emit("action", f"Iter{step_count+1} 调用工具: {parsed['action']} | 输入={str(parsed['action_input'])[:150]}")
                
                observation = await self._execute_tool(
                    parsed["action"],
                    parsed["action_input"],
                    state,
                )
                
                steps.append(f"Observation: {self._truncate_str(str(observation), _MAX_OBSERVATION_CHARS)}")
                _push_trace(
                    {
                        "type": "tool_call",
                        "iteration": step_count + 1,
                        "tool": parsed["action"],
                        "input": parsed["action_input"],
                        "observation": str(observation)[:1200],
                    }
                )
                self._emit("observation", f"Iter{step_count+1} 工具结果({parsed['action']}): {str(observation)[:200]}")
                evidence_graph["nodes"].append(
                    self._make_node(
                        node_id=f"obs_{step_count + 1}",
                        node_type="TIMELINE",
                        label=f"Observation {step_count + 1}",
                        attributes={
                            "tool": parsed["action"],
                            "observation": str(observation)[:400],
                            "iteration": step_count + 1,
                        },
                    )
                )
                self._enrich_evidence_graph_from_observation(
                    state=state,
                    evidence_graph=evidence_graph,
                    action=parsed["action"],
                    action_input=parsed["action_input"],
                    observation=observation,
                )
                confidence_candidate = self._estimate_confidence_from_observation(str(observation), before_conf)
                confidence_history.append(
                    {
                        "iteration": step_count + 1,
                        "confidence": confidence_candidate,
                        "confidence_delta": round(confidence_candidate - float(before_conf or 0.0), 6),
                        "tools_used": [parsed["action"]],
                        "new_evidence_count": 1,
                    }
                )
                self._emit("confidence", f"Iter{step_count+1} 置信度更新: {before_conf:.3f} → {confidence_candidate:.3f} (Δ={confidence_candidate - float(before_conf or 0.0):+.4f})")
                
                # 兼容原生 tool calling 的 message 格式
                if hasattr(response, "tool_calls") and response.tool_calls:
                    tool_call = response.tool_calls[0]
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": tool_call.get("function", {}).get("name"),
                        "content": str(observation)
                    })
                else:
                    messages.append(self._observation_message(f"Observation: {observation}"))
                    
                step_count += 1
            else:
                # CRASH-001: 即使没有有效操作，也要增加步数，避免死循环
                # 注意：不能添加虚假的用户消息
                self._emit("warn", f"Iter{step_count+1} 无效操作，步进并重试")
                step_count += 1

        final_answer = self._synthesize_final_answer(messages, steps)
        self._emit("warn", f"达到最大推理步骤({self._config.max_steps})，综合最终答案")
        if final_answer:
            confidence = self._extract_confidence(final_answer)
            root_cause = self._extract_root_cause(final_answer)
            steps.append(f"Thought: {final_answer}")
            return {
                "steps": steps,
                "confidence": confidence,
                "confidence_breakdown": {
                    "evidence_sufficiency": confidence,
                    "topology_consistency": confidence,
                    "temporal_consistency": confidence,
                    "historical_prior_match": confidence,
                    "aggregated_confidence": confidence,
                },
                "root_cause": root_cause,
                "confidence_history": confidence_history,
                "iteration_count": step_count + 1,
                "operator_trace": operator_trace,
                "internal_reasoning": internal_reasoning,
                "hypothesis_queue": hypothesis_queue,
                "hypothesis_candidates": [],
                "rejected_hypotheses": [],
                "evidence_graph": evidence_graph,
            }

        raise RCAError("达到最大推理步骤", step=step_count, recoverable=True)

    def _estimate_confidence_from_observation(self, observation: str, previous_confidence: float) -> float:
        normalized = str(observation or "").lower()
        delta = 0.04
        if "error" in normalized or "failed" in normalized:
            delta = -0.06
        if "success" in normalized or "reachable" in normalized or "found" in normalized:
            delta = 0.08
        return max(0.0, min(1.0, float(previous_confidence or 0.0) + delta))

    def _evaluate_root_cause_closure(self, state: DiagnosticState, steps: list[str], root_cause: str) -> dict[str, Any]:
        notes: list[str] = []
        max_confidence = 1.0
        root_text = str(root_cause or "").lower()
        analysis_text = "\n".join(steps).lower()
        has_query_success = "query_dtc" in analysis_text and ("success': true" in analysis_text or '"success": true' in analysis_text)
        has_query_failure = "query_dtc" in analysis_text and ("success': false" in analysis_text or '"success": false' in analysis_text)
        if not has_query_success:
            notes.append("缺少正向DTC查询证据")
            max_confidence = min(max_confidence, 0.62)
        focused = list(state.get("focused_dtcs", []))
        related = [d for d in focused if str(d.get("ecu_id", "")).upper() in root_cause.upper()]
        if not related:
            notes.append("根因未与明确ECU建立闭环映射")
            max_confidence = min(max_confidence, 0.58)
        if has_query_failure and "排除" not in analysis_text and "exclude" not in analysis_text:
            notes.append("存在查询失败但无排除链路说明")
            max_confidence = min(max_confidence, 0.55)
        if "泊车" in root_text and "communication" not in analysis_text and "通信" not in analysis_text:
            notes.append("泊车根因缺少通信证据支撑")
            max_confidence = min(max_confidence, 0.6)
        return {"passed": len(notes) == 0, "notes": notes, "max_confidence": max_confidence}

    def _pick_recall_seed_ecu(self, state: DiagnosticState) -> str | None:
        focused = list(state.get("focused_dtcs", []))
        if not focused:
            return None
        ranked = sorted(focused, key=lambda x: int(x.get("focus_score", x.get("priority_score", 0)) or 0), reverse=True)
        for item in ranked:
            ecu = str(item.get("ecu_id", "")).upper().strip()
            if ecu:
                return ecu
        return None

    def _calculate_confidence_breakdown(
        self,
        state: DiagnosticState,
        steps: list[str],
        root_cause: str,
        closure_check: dict[str, Any],
    ) -> dict[str, float]:
        text = "\n".join(steps).lower()
        focused_count = max(1, len(state.get("focused_dtcs", [])))
        query_success_count = text.count("success': true") + text.count('"success": true')
        query_fail_count = text.count("success': false") + text.count('"success": false')
        if query_success_count <= 0:
            evidence_sufficiency = 0.45
        else:
            evidence_sufficiency = min(1.0, 0.86 + 0.08 * min(query_success_count, 2) - 0.04 * min(query_fail_count, 2))
        topology = state.get("topology_graph") or {}
        topology_nodes = max(0, len(topology.get("nodes", [])))
        topology_edges = max(0, len(topology.get("edges", [])))
        topology_consistency = min(1.0, 0.82 + min(0.12, topology_edges / max(5, topology_nodes * 3)) + (0.04 if "ecu" in root_cause.lower() else 0.0))
        temporal_consistency = 0.9 if ("时间线" in root_cause or "timeout" in root_cause.lower() or "超时" in root_cause) else 0.8
        rag = state.get("rag_knowledge") or {}
        definitions = len(rag.get("dtc_definitions", [])) if isinstance(rag, dict) else 0
        historical_prior_match = min(1.0, 0.82 + min(0.12, definitions / 10))
        if not closure_check.get("passed"):
            evidence_sufficiency = min(evidence_sufficiency, 0.65)
            topology_consistency = min(topology_consistency, 0.7)
        if query_fail_count > 0 and "排除" not in text and "exclude" not in text:
            temporal_consistency = min(temporal_consistency, 0.6)
        aggregated_confidence = (
            evidence_sufficiency
            * topology_consistency
            * temporal_consistency
            * historical_prior_match
        )
        calibrated_confidence = max(0.0, min(0.97, aggregated_confidence * 1.25))
        return {
            "evidence_sufficiency": round(evidence_sufficiency, 6),
            "topology_consistency": round(topology_consistency, 6),
            "temporal_consistency": round(temporal_consistency, 6),
            "historical_prior_match": round(historical_prior_match, 6),
            "aggregated_confidence": round(calibrated_confidence, 6),
            "raw_product_confidence": round(max(0.0, min(1.0, aggregated_confidence)), 6),
        }

    def _build_hypothesis_outputs(
        self,
        *,
        state: DiagnosticState,
        root_cause: str,
        confidence_breakdown: dict[str, float],
        closure_check: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        focused = sorted(
            list(state.get("focused_dtcs", [])),
            key=lambda item: int(item.get("focus_score", item.get("priority_score", 0)) or 0),
            reverse=True,
        )
        candidates = []
        for idx, item in enumerate(focused[:3], start=1):
            ecu_id = str(item.get("ecu_id", "")).upper()
            candidates.append(
                {
                    "rank": idx,
                    "hypothesis": f"ECU {ecu_id or 'UNKNOWN'} 关联故障链路导致目标症状",
                    "supporting_dtc": item.get("code"),
                    "score": round(max(0.0, confidence_breakdown["aggregated_confidence"] - (idx - 1) * 0.12), 6),
                    "status": "selected" if ecu_id and ecu_id in root_cause.upper() else "candidate",
                    "related_ecu": ecu_id,
                }
            )
        rejected = []
        for item in focused[3:8]:
            rejected.append(
                {
                    "hypothesis": f"DTC {item.get('code')} 作为主因",
                    "reason": "优先级/证据一致性不足，暂不作为主根因",
                    "related_ecu": item.get("ecu_id"),
                }
            )
        if not closure_check.get("passed"):
            rejected.append(
                {
                    "hypothesis": "当前根因直接成立",
                    "reason": "; ".join(closure_check.get("notes", [])) or "证据闭环未通过",
                    "related_ecu": None,
                }
            )
        return candidates, rejected

    def _initialize_evidence_graph(self, state: DiagnosticState) -> dict[str, Any]:
        graph: dict[str, Any] = {"nodes": [], "edges": []}
        self._seed_symptom_nodes(state, graph)
        self._seed_timeline_nodes(state, graph)
        self._seed_dtc_and_ecu_nodes(state, graph)
        self._seed_topology_protocol_nodes(state, graph)
        self._seed_historical_case_nodes(state, graph)
        return graph

    def _seed_symptom_nodes(self, state: DiagnosticState, graph: dict[str, Any]) -> None:
        user_context = str(state.get("user_context", "") or "").strip()
        if not user_context:
            return
        self._upsert_node(
            graph,
            self._make_node(
                node_id="symptom:user_context",
                node_type="SYMPTOM",
                label="用户症状",
                attributes={"text": user_context},
            ),
        )

    def _seed_timeline_nodes(self, state: DiagnosticState, graph: dict[str, Any]) -> None:
        dtcs = list((state.get("structured_data") or {}).get("dtc_codes", []))
        active_count = sum(1 for item in dtcs if str(item.get("status", "")).lower() == "active")
        stored_count = sum(1 for item in dtcs if str(item.get("status", "")).lower() == "stored")
        freeze_frames = list((state.get("structured_data") or {}).get("freeze_frames", []))
        timeline_text = f"active_dtc={active_count}; stored_dtc={stored_count}; freeze_frames={len(freeze_frames)}"
        self._upsert_node(
            graph,
            self._make_node(
                node_id="timeline:report_state",
                node_type="TIMELINE",
                label="报告时序摘要",
                attributes={"summary": timeline_text},
            ),
        )

    def _seed_dtc_and_ecu_nodes(self, state: DiagnosticState, graph: dict[str, Any]) -> None:
        focused_dtcs = list(state.get("focused_dtcs", [])) or list((state.get("structured_data") or {}).get("dtc_codes", []))
        for item in focused_dtcs:
            code = str(item.get("code", "")).upper().strip()
            ecu_id = str(item.get("ecu_id", "")).upper().strip()
            if not code:
                continue
            dtc_node_id = f"dtc:{code}"
            self._upsert_node(
                graph,
                self._make_node(
                    node_id=dtc_node_id,
                    node_type="DTC",
                    label=code,
                    attributes={
                        "status": item.get("status"),
                        "severity": item.get("severity"),
                        "description": item.get("description"),
                        "priority_score": item.get("priority_score"),
                        "focus_score": item.get("focus_score"),
                    },
                ),
            )
            if ecu_id:
                ecu_node_id = f"ecu:{ecu_id}"
                self._upsert_node(
                    graph,
                    self._make_node(
                        node_id=ecu_node_id,
                        node_type="ECU",
                        label=ecu_id,
                        attributes={},
                    ),
                )
                self._upsert_edge(
                    graph,
                    source_id=dtc_node_id,
                    target_id=ecu_node_id,
                    relation="depends_on",
                    evidence="DTC归属该ECU",
                    attributes={"source": "focused_dtcs"},
                )

    def _seed_topology_protocol_nodes(self, state: DiagnosticState, graph: dict[str, Any]) -> None:
        topology = state.get("topology_graph") or {}
        for node in topology.get("nodes", []):
            ecu_id = str(node.get("ecu_id", "")).upper().strip()
            if not ecu_id:
                continue
            self._upsert_node(
                graph,
                self._make_node(
                    node_id=f"ecu:{ecu_id}",
                    node_type="ECU",
                    label=ecu_id,
                    attributes={
                        "domain": node.get("domain"),
                        "power_segment": node.get("power_segment"),
                        "gateway": node.get("gateway"),
                    },
                ),
            )
        for edge in topology.get("edges", []):
            source_id = str(edge.get("source_id", "")).upper().strip()
            target_id = str(edge.get("target_id", "")).upper().strip()
            protocol = str(edge.get("protocol", "")).upper().strip()
            if not source_id or not target_id or not protocol:
                continue
            protocol_node_id = f"protocol:{protocol}"
            self._upsert_node(
                graph,
                self._make_node(
                    node_id=protocol_node_id,
                    node_type="PROTOCOL",
                    label=protocol,
                    attributes={},
                ),
            )
            self._upsert_edge(
                graph,
                source_id=f"ecu:{source_id}",
                target_id=protocol_node_id,
                relation="depends_on",
                evidence="拓扑边源端依赖该通信协议",
                attributes={"direction": "source"},
            )
            self._upsert_edge(
                graph,
                source_id=protocol_node_id,
                target_id=f"ecu:{target_id}",
                relation="depends_on",
                evidence="拓扑边目标端通过该通信协议可达",
                attributes={"direction": "target"},
            )

    def _seed_historical_case_nodes(self, state: DiagnosticState, graph: dict[str, Any]) -> None:
        rag_knowledge = state.get("rag_knowledge") or {}
        case_studies = rag_knowledge.get("case_studies", []) if isinstance(rag_knowledge, dict) else []
        for index, item in enumerate(case_studies[:5], start=1):
            label = str(item.get("title") or item.get("summary") or f"历史案例{index}")
            self._upsert_node(
                graph,
                self._make_node(
                    node_id=f"history:{index}",
                    node_type="HISTORICAL_CASE",
                    label=label,
                    attributes=item if isinstance(item, dict) else {"raw": str(item)},
                ),
            )

    def _finalize_evidence_graph(
        self,
        *,
        state: DiagnosticState,
        evidence_graph: dict[str, Any],
        root_cause: str,
        hypothesis_candidates: list[dict[str, Any]],
        rejected_hypotheses: list[dict[str, Any]],
        closure_check: dict[str, Any],
    ) -> None:
        root_cause_upper = root_cause.upper()
        symptom_node_id = "symptom:user_context"
        timeline_node_id = "timeline:report_state"
        selected_candidate_id = None
        for item in hypothesis_candidates:
            rank = item.get("rank")
            node_id = f"candidate:{rank}"
            self._upsert_node(
                evidence_graph,
                self._make_node(
                    node_id=node_id,
                    node_type="ROOT_CAUSE_CANDIDATE",
                    label=str(item.get("hypothesis", "")),
                    attributes={
                        "rank": rank,
                        "score": item.get("score"),
                        "status": item.get("status"),
                        "supporting_dtc": item.get("supporting_dtc"),
                        "related_ecu": item.get("related_ecu"),
                    },
                ),
            )
            if item.get("status") == "selected":
                selected_candidate_id = node_id
            dtc_code = str(item.get("supporting_dtc", "")).upper().strip()
            related_ecu = str(item.get("related_ecu", "")).upper().strip()
            if dtc_code:
                self._upsert_edge(
                    evidence_graph,
                    source_id=f"dtc:{dtc_code}",
                    target_id=node_id,
                    relation="supports",
                    evidence="候选根因由高优先级DTC支持",
                    attributes={},
                )
            if related_ecu:
                self._upsert_edge(
                    evidence_graph,
                    source_id=node_id,
                    target_id=f"ecu:{related_ecu}",
                    relation="depends_on",
                    evidence="根因候选依赖相关ECU",
                    attributes={},
                )
        if selected_candidate_id:
            if self._node_exists(evidence_graph, symptom_node_id):
                self._upsert_edge(
                    evidence_graph,
                    source_id=symptom_node_id,
                    target_id=selected_candidate_id,
                    relation="supports",
                    evidence="用户症状与根因候选相符",
                    attributes={},
                )
            if self._node_exists(evidence_graph, timeline_node_id):
                self._upsert_edge(
                    evidence_graph,
                    source_id=timeline_node_id,
                    target_id=selected_candidate_id,
                    relation="supports",
                    evidence="Active/Stored DTC时序支持该根因",
                    attributes={},
                )
            for node in evidence_graph["nodes"]:
                if node.get("node_type") == "HISTORICAL_CASE":
                    self._upsert_edge(
                        evidence_graph,
                        source_id=node["id"],
                        target_id=selected_candidate_id,
                        relation="supports",
                        evidence="历史案例与当前根因候选相似",
                        attributes={},
                    )
        for index, item in enumerate(rejected_hypotheses, start=1):
            rejected_id = f"candidate:rejected:{index}"
            self._upsert_node(
                evidence_graph,
                self._make_node(
                    node_id=rejected_id,
                    node_type="ROOT_CAUSE_CANDIDATE",
                    label=str(item.get("hypothesis", "")),
                    attributes={"status": "rejected", "reason": item.get("reason"), "related_ecu": item.get("related_ecu")},
                ),
            )
            related_ecu = str(item.get("related_ecu", "")).upper().strip()
            if related_ecu and self._node_exists(evidence_graph, f"ecu:{related_ecu}"):
                self._upsert_edge(
                    evidence_graph,
                    source_id=f"ecu:{related_ecu}",
                    target_id=rejected_id,
                    relation="excluded_by",
                    evidence=str(item.get("reason", "")),
                    attributes={},
                )
            elif self._node_exists(evidence_graph, timeline_node_id):
                self._upsert_edge(
                    evidence_graph,
                    source_id=timeline_node_id,
                    target_id=rejected_id,
                    relation="excluded_by",
                    evidence=str(item.get("reason", "")),
                    attributes={},
                )
        if selected_candidate_id and closure_check.get("notes"):
            for note in closure_check.get("notes", []):
                self._upsert_edge(
                    evidence_graph,
                    source_id=timeline_node_id,
                    target_id=selected_candidate_id,
                    relation="contradicts",
                    evidence=str(note),
                    attributes={},
                )
        if selected_candidate_id:
            for node in evidence_graph["nodes"]:
                if node.get("node_type") != "DTC":
                    continue
                description = _safe_lower(node.get("attributes", {}).get("description"))
                label = _safe_lower(node.get("label"))
                if "氧" in description or "oxygen" in description or label.startswith("p014"):
                    self._upsert_edge(
                        evidence_graph,
                        source_id=node["id"],
                        target_id=selected_candidate_id,
                        relation="contradicts",
                        evidence="该DTC更偏动力系统，和泊车主因弱相关",
                        attributes={"reason": "domain_mismatch"},
                    )

    def _enrich_evidence_graph_from_observation(
        self,
        *,
        state: DiagnosticState,
        evidence_graph: dict[str, Any],
        action: str,
        action_input: dict[str, Any],
        observation: Any,
    ) -> None:
        if action == "query_dtc":
            dtc_code = str(action_input.get("dtc_code", "")).upper().strip()
            result = observation if isinstance(observation, dict) else {}
            if dtc_code:
                self._upsert_node(
                    evidence_graph,
                    self._make_node(
                        node_id=f"dtc:{dtc_code}",
                        node_type="DTC",
                        label=dtc_code,
                        attributes={"queried": True},
                    ),
                )
            if result.get("success"):
                for item in result.get("results", [])[:3]:
                    ecu_id = str(item.get("ecu_id", "")).upper().strip()
                    matched_code = str(item.get("dtc_code", "")).upper().strip() or dtc_code
                    if matched_code:
                        self._upsert_node(
                            evidence_graph,
                            self._make_node(
                                node_id=f"dtc:{matched_code}",
                                node_type="DTC",
                                label=matched_code,
                                attributes={"description": item.get("description"), "matched_by_query": dtc_code},
                            ),
                        )
                    if ecu_id:
                        self._upsert_node(
                            evidence_graph,
                            self._make_node(
                                node_id=f"ecu:{ecu_id}",
                                node_type="ECU",
                                label=ecu_id,
                                attributes={"ecu_shortname": item.get("ecu_shortname")},
                            ),
                        )
                    if matched_code and ecu_id:
                        self._upsert_edge(
                            evidence_graph,
                            source_id=f"dtc:{matched_code}",
                            target_id=f"ecu:{ecu_id}",
                            relation="depends_on",
                            evidence="DTC查询结果表明该故障码映射到该ECU",
                            attributes={"query_code": dtc_code},
                        )
            elif dtc_code:
                self._upsert_edge(
                    evidence_graph,
                    source_id="timeline:report_state",
                    target_id=f"dtc:{dtc_code}",
                    relation="contradicts",
                    evidence="数据库未命中该DTC，需要排除或补充证据",
                    attributes={"query_code": dtc_code},
                )
        elif action == "check_ecu_hop_distance":
            result = observation if isinstance(observation, dict) else {}
            source_ecu = str(result.get("source_ecu") or action_input.get("source_ecu") or "").upper().strip()
            target_ecu = str(result.get("target_ecu") or action_input.get("target_ecu") or "").upper().strip()
            for edge in result.get("path_edges", []) or []:
                for protocol in edge.get("protocols", []) or []:
                    protocol_name = str(protocol).upper().strip()
                    if not protocol_name:
                        continue
                    protocol_node_id = f"protocol:{protocol_name}"
                    self._upsert_node(
                        evidence_graph,
                        self._make_node(
                            node_id=protocol_node_id,
                            node_type="PROTOCOL",
                            label=protocol_name,
                            attributes={"source": "hop_distance"},
                        ),
                    )
                    from_ecu = str(edge.get("from_ecu", "")).upper().strip()
                    to_ecu = str(edge.get("to_ecu", "")).upper().strip()
                    if from_ecu:
                        self._upsert_edge(
                            evidence_graph,
                            source_id=f"ecu:{from_ecu}",
                            target_id=protocol_node_id,
                            relation="depends_on",
                            evidence="跳距分析识别到该协议链路",
                            attributes={},
                        )
                    if to_ecu:
                        self._upsert_edge(
                            evidence_graph,
                            source_id=protocol_node_id,
                            target_id=f"ecu:{to_ecu}",
                            relation="depends_on",
                            evidence="协议链路可达目标ECU",
                            attributes={},
                        )
            if source_ecu and target_ecu and result.get("reachable"):
                self._upsert_edge(
                    evidence_graph,
                    source_id=f"ecu:{source_ecu}",
                    target_id=f"ecu:{target_ecu}",
                    relation="supports",
                    evidence=f"跳距分析显示 {source_ecu}->{target_ecu} 可达",
                    attributes={"hop_distance": result.get("hop_distance")},
                )

    def _make_node(self, *, node_id: str, node_type: str, label: str, attributes: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": node_id,
            "node_type": node_type,
            "label": label,
            "attributes": attributes,
        }

    def _upsert_node(self, evidence_graph: dict[str, Any], node: dict[str, Any]) -> None:
        for existing in evidence_graph["nodes"]:
            if existing.get("id") == node["id"]:
                existing.setdefault("attributes", {}).update(node.get("attributes", {}))
                if node.get("label"):
                    existing["label"] = node["label"]
                if node.get("node_type"):
                    existing["node_type"] = node["node_type"]
                return
        evidence_graph["nodes"].append(node)

    def _upsert_edge(
        self,
        evidence_graph: dict[str, Any],
        *,
        source_id: str,
        target_id: str,
        relation: str,
        evidence: str,
        attributes: dict[str, Any],
    ) -> None:
        if not self._node_exists(evidence_graph, source_id) or not self._node_exists(evidence_graph, target_id):
            return
        for existing in evidence_graph["edges"]:
            if (
                existing.get("source_id") == source_id
                and existing.get("target_id") == target_id
                and existing.get("relation") == relation
                and existing.get("evidence") == evidence
            ):
                existing.setdefault("attributes", {}).update(attributes)
                return
        evidence_graph["edges"].append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "relation": relation,
                "evidence": evidence,
                "attributes": attributes,
            }
        )

    def _node_exists(self, evidence_graph: dict[str, Any], node_id: str) -> bool:
        return any(item.get("id") == node_id for item in evidence_graph.get("nodes", []))

    @staticmethod
    def _truncate_str(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."

    @staticmethod
    def _summarize_dtc_observation(obs_text: str) -> str:
        try:
            data = json.loads(obs_text)
        except (json.JSONDecodeError, TypeError):
            return RCAEngine._truncate_str(obs_text, _MAX_OBSERVATION_CHARS)
        if not isinstance(data, dict):
            return RCAEngine._truncate_str(obs_text, _MAX_OBSERVATION_CHARS)
        success = data.get("success")
        results = data.get("results", [])
        if not isinstance(results, list) or not results:
            return json.dumps({"success": success, "results": []}, ensure_ascii=False)
        lean_results = []
        for r in results[:3]:
            lean_results.append({
                "dtc_code": r.get("dtc_code"),
                "description": r.get("description"),
                "dtc_level": r.get("dtc_level"),
                "ecu_id": r.get("ecu_id"),
                "ecu_longname": r.get("ecu_longname"),
            })
        lean = {"success": success, "result_count": len(results), "results": lean_results}
        summary = json.dumps(lean, ensure_ascii=False)
        if len(summary) > _MAX_OBSERVATION_CHARS:
            return summary[:_MAX_OBSERVATION_CHARS - 3] + "..."
        return summary

    async def _preload_observations(self, state: DiagnosticState) -> tuple[list[str], list[dict]]:
        steps: list[str] = []
        messages: list[dict] = []
        dtcs = state.get("structured_data", {}).get("dtc_codes", [])
        if not dtcs:
            return steps, messages

        sorted_dtcs = sorted(dtcs, key=lambda x: x.get("priority_score", 0), reverse=True)
        sampled = sorted_dtcs[:3]

        for item in sampled:
            raw = item.get("raw_data", {}) or {}
            query_code = (raw.get("fehler_typ") or item.get("code") or "").strip()
            if not query_code:
                continue
            observation = await self._execute_tool("query_dtc", {"dtc_code": query_code, "limit": 2}, state)
            obs_str = str(observation)
            lean_obs = self._summarize_dtc_observation(obs_str)
            steps.append(f"Observation: preload query_dtc({query_code}) => {lean_obs}")
            messages.append(self._observation_message(f"Observation: preload query_dtc({query_code}) => {observation}"))

        ecu_ids = [d.get("ecu_id") for d in sampled if d.get("ecu_id")]
        if len(ecu_ids) >= 2:
            source_ecu = str(ecu_ids[0]).upper()
            target_ecu = str(ecu_ids[1]).upper()
            observation = await self._execute_tool(
                "check_ecu_hop_distance",
                {"source_ecu": source_ecu, "target_ecu": target_ecu, "max_hops": 4},
                state
            )
            obs_str = str(observation)
            lean_obs = self._truncate_str(obs_str, _MAX_OBSERVATION_CHARS)
            steps.append(f"Observation: preload check_ecu_hop_distance({source_ecu},{target_ecu}) => {lean_obs}")
            messages.append(
                self._observation_message(
                    f"Observation: preload check_ecu_hop_distance({source_ecu},{target_ecu}) => {observation}"
                )
            )
        return steps, messages
    
    def _build_user_prompt(self, state: DiagnosticState) -> str:
        # 检查是否包含审计失败反馈，如果有，则可能需要注入额外的上下文
        audit_log = state.get("audit_log", [])
        if not state.get("audit_passed", True) and audit_log:
            # 简化处理，实际应该调用注册表中的不同模板或传入参数
            pass

        return self._prompt_registry.get(
            "rca",
            dtc_codes=self._prepare_dtc_payload(state["structured_data"].get("dtc_codes", [])),
            topology_graph=self._prepare_topology_payload(state.get("topology_graph")),
            case_studies=(state.get("rag_knowledge") or {}).get("case_studies", []),
            user_context=state.get("user_context", ""),
            human_hint=state.get("human_hint"),
            audit_log=state.get("audit_log", [])
        )

    def _prepare_dtc_payload(self, dtc_codes: list[dict]) -> list[dict]:
        payload = []
        for dtc in dtc_codes:
            raw = dtc.get("raw_data", {}) or {}
            payload.append({
                "code": dtc.get("code"),
                "status": dtc.get("status"),
                "severity": dtc.get("severity"),
                "priority_score": dtc.get("priority_score"),
                "event_type": dtc.get("event_type") or raw.get("fault_event"),
                "fault_status": raw.get("fault_status"),
                "dtc_priority": raw.get("dtc_priority"),
                "fehler_typ": raw.get("fehler_typ"),
                "description": dtc.get("description"),
                "ecu_id": dtc.get("ecu_id"),
            })
        return payload

    def _prepare_topology_payload(self, topology_graph: dict | None) -> dict | None:
        if not topology_graph:
            return topology_graph
        nodes = []
        for node in topology_graph.get("nodes", []):
            item = {
                "ecu_id": node.get("ecu_id"),
                "domain": node.get("domain"),
                "status": node.get("status"),
                "communication_status": node.get("communication_status"),
            }
            if node.get("power_segment") not in (None, "", "Unknown"):
                item["power_segment"] = node.get("power_segment")
            if node.get("gateway"):
                item["gateway"] = True
            nodes.append(item)
        return {
            **topology_graph,
            "nodes": nodes
        }
    
    def _get_available_tools(self, state: dict) -> list[dict]:
        custom_tools = [
            {
                "type": "function",
                "function": {
                    "name": "recall_deferred_dtcs",
                    "description": "根据ECU从deferred_dtcs中回捞潜在关联DTC",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ecu_id": {"type": "string"},
                            "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                        },
                        "required": ["ecu_id"],
                    },
                },
            }
        ]
        if self._disclosure:
            confidence = state.get("confidence_score", 0.0)
            previous_level = self._disclosure._current_level
            new_level = self._disclosure.update_level(confidence, state)
            self._last_disclosure_event = {
                "previous_level": previous_level,
                "new_level": new_level,
                "reason": f"confidence={float(confidence or 0.0):.4f}",
            }
            return self._disclosure.get_available_tools() + custom_tools
        
        self._last_disclosure_event = None
        return self._tool_registry.list_openai_tools() + custom_tools
    
    def _parse_react_response(self, response: str) -> dict:
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", response, re.DOTALL)
        action_match = re.search(r"Action:\s*(\w+)", response)
        action_input_match = re.search(r"Action Input:\s*(\{.+?\}|\[.+?\]|\".+?\")", response, re.DOTALL)
        final_answer_match = re.search(r"Final Answer:\s*(.+)$", response, re.DOTALL)
        
        if not final_answer_match:
            final_answer_match = re.search(r"根因:\s*(.+)$", response, re.DOTALL)
        
        action_input = None
        if action_input_match:
            try:
                action_input = json.loads(action_input_match.group(1))
            except json.JSONDecodeError:
                action_input = {"query": action_input_match.group(1).strip('"')}
        
        return {
            "thought": thought_match.group(1).strip() if thought_match else response.strip(),
            "action": action_match.group(1) if action_match else None,
            "action_input": action_input,
            "final_answer": final_answer_match.group(1).strip() if final_answer_match else None,
        }
    
    async def _execute_tool(self, action: str, action_input: dict, state: dict) -> str:
        if action == "recall_deferred_dtcs":
            ecu_id = str(action_input.get("ecu_id", "")).upper()
            limit = max(1, min(20, int(action_input.get("limit", 5) or 5)))
            deferred = list(state.get("deferred_dtcs", []))
            recalled = [item for item in deferred if str(item.get("ecu_id", "")).upper() == ecu_id][:limit]
            return json.dumps(
                {
                    "success": True,
                    "ecu_id": ecu_id,
                    "recalled_count": len(recalled),
                    "recalled_dtcs": recalled,
                },
                ensure_ascii=False,
            )
        # 1. 注入上下文
        context = {
            "state": state,
            "session_id": state.get("session_id"),
            "user_id": state.get("user_id"),
        }
        
        # 2. 执行工具
        try:
            wrapped_handler = self._tool_registry.inject_context_to_handler(action, **context)
            result = wrapped_handler(**action_input)
            
            # Handle async tools
            if asyncio.iscoroutine(result):
                result = await result
                
            return str(result)
        except asyncio.CancelledError:
            logger.warning(f"[RCAEngine] Tool {action} was cancelled")
            raise  # 重新抛出，不吞掉取消信号
        except Exception as e:
            logger.error(f"[RCAEngine] 工具执行失败 {action}: {e}")
            return f"Error executing tool {action}: {str(e)}"
    
    def _extract_confidence(self, final_answer: str) -> float:
        confidence_match = re.search(r"(?:confidence|置信度)[:\s]+(\d+\.?\d*)", final_answer, re.I)
        
        if confidence_match:
            confidence = float(confidence_match.group(1))
            return max(0.0, min(1.0, confidence))
        
        return 0.5
    
    def _extract_root_cause(self, final_answer: str) -> str:
        cause_match = re.search(r"(?:root cause|根因)[:\s]+(.+?)(?=(?:confidence|置信度)|$)", final_answer, re.I | re.DOTALL)
        
        if cause_match:
            return cause_match.group(1).strip()
        
        return final_answer.strip()

    def _synthesize_final_answer(self, messages: list[dict], steps: list[str]) -> str | None:
        prompt = {
            "role": "user",
            "content": "请基于已有 Observation 给出最终结论，不再调用工具。格式必须包含“根因:”和“置信度:”。"
        }
        synth_messages = messages + [prompt]
        try:
            response = self._llm.generate(messages=synth_messages, temperature=0.1)
            text = (response.content or "").strip()
            return text or None
        except Exception as e:
            logger.warning(f"[RCAEngine] 最终结论综合失败: {e}")
            return None
