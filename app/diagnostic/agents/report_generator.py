"""Report Generator - 报告生成 Agent

负责将所有诊断信息整合，生成最终的结构化诊断报告。

依赖:
    - diagnostic.state
    - app.providers (LLM)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from app.diagnostic.state import DiagnosticState, FinalReport
from app.diagnostic.prompts.registry import PromptRegistry

if TYPE_CHECKING:
    from app.providers import LLMProvider

logger = logger.bind(component="ReportGenerator")

_MAX_DTC_DEFINITION_CHARS = 480
_MAX_OBSERVATION_CHARS = 600
_MAX_ANALYSIS_STEP_CHARS = 800


class ReportGenerator:
    """最终报告生成 Agent"""

    NODE_NAME = "report_generator"

    def __init__(self, llm: "LLMProvider", prompt_registry: PromptRegistry, config: dict | None = None):
        self._llm = llm
        self._prompt_registry = prompt_registry
        self._config = config or {}
        logger.info("[ReportGenerator] 初始化完成")

    def generate(self, state: DiagnosticState) -> DiagnosticState:
        """生成最终报告

        Args:
            state: 当前诊断状态

        Returns:
            更新后的诊断状态
        """
        logger.info("[ReportGenerator] 开始生成报告")

        system_prompt = self._prompt_registry.get_system_prompt("report")

        lean_structured = self._lean_structured_data(state.get("structured_data", {}))
        lean_analysis = self._lean_analysis_steps(state.get("analysis_steps", []))
        lean_dtc_defs = self._lean_dtc_definitions_from_rag(state.get("rag_knowledge"))

        user_prompt = self._prompt_registry.get(
            "report",
            structured_data=json.dumps(lean_structured, ensure_ascii=False),
            analysis_steps=json.dumps(lean_analysis, ensure_ascii=False),
            dtc_definitions=json.dumps(lean_dtc_defs, ensure_ascii=False),
            root_cause=state.get("preliminary_root_cause", ""),
            confidence_score=state.get("confidence_score", 0.0),
            topology_graph=json.dumps(self._lean_topology(state.get("topology_graph", {})), ensure_ascii=False),
            case_studies=json.dumps((state.get("rag_knowledge") or {}).get("case_studies", []), ensure_ascii=False),
            audit_log=json.dumps(state.get("audit_log", []), ensure_ascii=False),
            risk_flags=json.dumps(state.get("risk_flags", []), ensure_ascii=False),
        )

        prompt_len = sum(len(str(m.get("content", ""))) for m in [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        logger.info(f"[ReportGenerator] Prompt 预估大小: {prompt_len} 字符 / ~{max(1, prompt_len // 4)} tokens")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self._llm.generate(messages=messages, temperature=0.1)
            markdown_report = response.content
            
            # 从生成的 markdown 中尝试提取一些结构化字段用于 FinalReport，或者生成一个结构化的对象
            # 简化实现：我们可以从原始数据和生成的报告中组装 FinalReport
            
            structured_data = state.get("structured_data", {})
            affected_ecus = self._build_affected_ecus(state)
            
            # 提取风险和建议（简单解析）
            recommendations = self._extract_section(markdown_report, "维修建议")
            risks = self._extract_section(markdown_report, "风险提示")
            
           # CHAIN-003: 添加冲突信息到报告中
            risks_with_conflicts = list(risks)
            if state.get("conflict_ignored"):
                conflicts = state.get("conflicts", [])
                risks_with_conflicts.append(f"⚠️ 用户已忽略审计冲突: {'; '.join(conflicts)}")
            
            # 确保 risk_flags 也被记录
            risk_flags = state.get("risk_flags", [])
            if risk_flags:
                risks_with_conflicts.extend([f"风险标记: {flag}" for flag in risk_flags])
                
            final_report = FinalReport(
                root_cause=state.get("preliminary_root_cause", ""),
                confidence_score=state.get("confidence_score", 0.0),
                affected_ecus=affected_ecus,
                evidence=state.get("analysis_steps", []) + [
                    f"TRANSITION: {item.get('node')} => {item.get('result')}"
                    for item in state.get("transition_log", [])[-10:]
                ],
                recommendations=recommendations,
                risks=risks_with_conflicts,
                audit_trail=state.get("audit_log", []),
                human_decisions=state.get("hitl_feedbacks", []),
                pending_confirmation=False,
                confidence_breakdown=state.get("confidence_breakdown", {}),
                hypothesis_candidates=state.get("hypothesis_candidates", []),
                rejected_hypotheses=state.get("rejected_hypotheses", []),
                evidence_graph=state.get("evidence_graph", {"nodes": [], "edges": []}),
            )
            
            # 可以将生成的 markdown 附加到最终结果中（这里使用一个特殊的字段或放到 state 的某个地方，虽然 TypedDict 没定义 markdown 字段，但可以附加到风险里或者直接保存）
            # 注意：实际生产中可能需要将 markdown 存入专门的存储或扩展 FinalReport
            
            return {
                **state,
                "final_report": final_report,
                "report_markdown": markdown_report,
                "follow_up_question": "报告已生成。是否继续分析新的电检报告，或基于当前报告换一个方向深挖（如通信链路/时间线/特定ECU）？",
                "conversation_suggestion": [
                    "继续分析新报告",
                    "对当前报告做通信链路深挖",
                    "对当前报告做拓扑关联深挖",
                ],
                "operator_trace": state.get("operator_trace", []),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": 0,
                "current_node": self.NODE_NAME,
            }
            
        except Exception as e:
            logger.exception(f"[ReportGenerator] 报告生成失败: {e}")
            
            # FUNC-003: 报告生成失败时提供降级回退
            structured_data = state.get("structured_data", {})
            affected_ecus = [ecu.get("ecu_id", "Unknown") for ecu in structured_data.get("ecu_states", [])]
            
            fallback_report = FinalReport(
                root_cause=state.get("preliminary_root_cause", "未知"),
                confidence_score=state.get("confidence_score", 0.0),
                affected_ecus=self._build_affected_ecus(state),
                evidence=state.get("analysis_steps", []),
                recommendations=["自动报告生成失败，请参考分析步骤进行人工判断"],
                risks=[f"报告生成服务异常: {str(e)}", "结果可能不完整"],
                audit_trail=state.get("audit_log", []),
                human_decisions=state.get("hitl_feedbacks", []),
                pending_confirmation=True,
                confidence_breakdown=state.get("confidence_breakdown", {}),
                hypothesis_candidates=state.get("hypothesis_candidates", []),
                rejected_hypotheses=state.get("rejected_hypotheses", []),
                evidence_graph=state.get("evidence_graph", {"nodes": [], "edges": []}),
            )
            
            return {
                **state,
                "final_report": fallback_report,
                "error_message": f"报告生成失败，已提供简化版: {str(e)}",
                "exit_code": 1, # Keep exit code 1 to indicate warning/partial failure? Or 0 for success with warning? Report says "output missing", so fallback provides output. Let's keep 1 or change to 0? The report says "exit_code: 1" in the bug description. I'll keep it as 1 to signal "something went wrong" but provide the report.
                "current_node": self.NODE_NAME,
            }

    def _lean_structured_data(self, structured_data: dict) -> dict:
        lean = dict(structured_data)
        lean["dtc_codes"] = [
            {
                k: v for k, v in dtc.items()
                if k != "raw_data"
            }
            for dtc in lean.get("dtc_codes", [])
        ]
        return lean

    def _lean_analysis_steps(self, analysis_steps: list) -> list:
        result: list = []
        for step in analysis_steps:
            if not isinstance(step, str):
                result.append(step)
                continue
            if len(step) > _MAX_ANALYSIS_STEP_CHARS:
                truncated = step[:_MAX_ANALYSIS_STEP_CHARS - 3] + "..."
                result.append(truncated)
            else:
                result.append(step)
        return result

    def _lean_dtc_definitions_from_rag(self, rag_knowledge) -> list:
        if not rag_knowledge:
            return []
        raw_defs = rag_knowledge.get("dtc_definitions", []) if isinstance(rag_knowledge, dict) else []
        if not raw_defs:
            return []
        lean = []
        for item in raw_defs:
            code = item.get("code", "")
            definition = item.get("definition", "")
            if len(definition) > _MAX_DTC_DEFINITION_CHARS:
                trunc_def = definition[:_MAX_DTC_DEFINITION_CHARS - 3] + "..."
            else:
                trunc_def = definition
            lean.append({"code": code, "definition": trunc_def, "source": item.get("source", "")})
        return lean

    def _lean_topology(self, topology_graph: dict) -> dict:
        if not topology_graph:
            return {}
        lean = dict(topology_graph)
        lean["nodes"] = [
            {
                "ecu_id": n.get("ecu_id"),
                "domain": n.get("domain"),
                "status": n.get("status"),
            }
            for n in lean.get("nodes", [])
        ]
        return lean

    def _extract_root_cause_ecus(self, state: DiagnosticState) -> set[str]:
        root_cause = str(state.get("preliminary_root_cause", "") or "")
        mentioned = set(re.findall(r"\b[0-9A-F]{4}\b", root_cause.upper()))
        if mentioned:
            return mentioned
        focused_dtcs = list(state.get("focused_dtcs", []))
        if not focused_dtcs:
            focused_dtcs = list((state.get("structured_data") or {}).get("dtc_codes", []))
        prioritized = sorted(
            focused_dtcs,
            key=lambda item: int(item.get("focus_score", item.get("priority_score", 0)) or 0),
            reverse=True,
        )
        selected = {str(item.get("ecu_id", "")).upper() for item in prioritized[:2] if str(item.get("ecu_id", "")).strip()}
        return selected

    def _build_affected_ecus(self, state: DiagnosticState) -> list[str]:
        root_ecus = self._extract_root_cause_ecus(state)
        if not root_ecus:
            return []
        topology = state.get("topology_graph") or {}
        edges = list(topology.get("edges", []))
        one_hop = set(root_ecus)
        for edge in edges:
            source = str(edge.get("source_id", "")).upper()
            target = str(edge.get("target_id", "")).upper()
            if source in root_ecus and target:
                one_hop.add(target)
            if target in root_ecus and source:
                one_hop.add(source)
        return sorted(one_hop)

    def _extract_section(self, text: str, section_name: str) -> list[str]:
        """简单的 Markdown 章节提取"""
        lines = text.split('\n')
        extracted = []
        in_section = False
        for line in lines:
            if line.startswith('## ') and section_name in line:
                in_section = True
                continue
            elif line.startswith('## ') and in_section:
                break
            
            if in_section and line.strip():
                # 移除列表符号
                clean_line = re.sub(r'^[-*0-9.]+\s+', '', line.strip())
                extracted.append(clean_line)
                
        return extracted if extracted else ["请参考完整诊断报告"]
