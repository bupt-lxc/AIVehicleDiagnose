"""Auditor Agent - 逻辑审计员

负责校验推理逻辑的正确性和一致性，包括物理逻辑、DTC 覆盖率、拓扑一致性等。

依赖:
    - diagnostic.state
    - app.providers (LLM)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from app.diagnostic.state import DiagnosticState

if TYPE_CHECKING:
    from app.providers import LLMProvider
    from app.diagnostic.prompts.registry import PromptRegistry

logger = logger.bind(component="AuditorAgent")


@dataclass(slots=True)
class AuditCheckResult:
    """审计检查结果"""
    check_name: str
    passed: bool
    log: list[str]
    conflict: str | None = None
    severity: str = "error"


class AuditorAgent:
    """逻辑审计 Agent"""
    
    NODE_NAME = "auditor"
    
    CRITICAL_CONFLICTS = [
        "physical_logic",
        "topology_consistency",
        "dtc_coverage",
    ]
    
    def __init__(
        self,
        llm: "LLMProvider | None" = None,
        prompt_registry: "PromptRegistry | None" = None,
        config: dict | None = None
    ):
        self._llm = llm
        self._prompt_registry = prompt_registry
        self._config = config or {}
        logger.info("[AuditorAgent] 初始化完成")
    
    def audit(self, state: DiagnosticState) -> DiagnosticState:
        """执行逻辑审计
        
        Args:
            state: 当前诊断状态
            
        Returns:
            更新后的诊断状态
        """
        audit_log = []
        conflicts = []
        
        if state.get("conflict_ignored"):
            logger.info("[AuditorAgent] 冲突已被用户忽略，跳过审计")
            audit_log.append("信息: 用户已选择忽略之前的审计冲突，跳过重复审计")
            return {
                **state,
                "audit_log": audit_log,
                "audit_passed": True,
                "conflicts": [],
                "current_node": self.NODE_NAME,
            }
        
        analysis_steps = state.get("analysis_steps", [])
        if not analysis_steps:
            logger.warning("[AuditorAgent] 无推理步骤可审计")
            audit_log.append("警告: 无推理步骤，跳过审计")
            
            return {
                **state,
                "audit_log": audit_log,
                "audit_passed": False,
                "conflicts": ["无推理步骤"],
                "current_node": self.NODE_NAME,
            }
        
        checks = [
            self._check_physical_logic(state),
            self._check_dtc_coverage(state),
            self._check_topology_consistency(state),
            self._check_evidence_quality(state),
            self._check_root_cause_closure(state),
            self._check_confidence_threshold(state),
        ]
        
        for result in checks:
            audit_log.extend(result.log)
            
            if not result.passed and result.conflict:
                conflicts.append(result.conflict)
                
                if result.severity == "error":
                    logger.error(f"[AuditorAgent] 检查失败 [{result.check_name}]: {result.conflict}")
        
        audit_passed = len(conflicts) == 0
        
        if audit_passed:
            logger.info("[AuditorAgent] 所有检查通过")
        else:
            logger.warning(f"[AuditorAgent] 发现 {len(conflicts)} 个冲突")
        
        result = {
            **state,
            "audit_log": audit_log,
            "audit_passed": audit_passed,
            "conflicts": conflicts,
            "current_node": self.NODE_NAME,
        }

        if not audit_passed:
            result["exit_code"] = 6  # Audit Failure

        return result
    
    def _check_physical_logic(self, state: DiagnosticState) -> AuditCheckResult:
        result = AuditCheckResult(
            check_name="physical_logic",
            passed=True,
            log=[],
        )
        
        root_cause = state.get("preliminary_root_cause", "")
        
        if not root_cause:
            result.passed = False
            result.conflict = "根因为空"
            result.log.append("错误: 根因为空，无法进行物理逻辑检查")
            return result
        
        power_keywords = ["断电", "供电", "电源", "power", "voltage"]
        communication_keywords = ["丢失通讯", "通信", "无法通讯", "communication lost"]
        
        has_power_issue = any(kw in root_cause.lower() for kw in power_keywords)
        has_comm_issue = any(kw in root_cause.lower() for kw in communication_keywords)
        
        if has_power_issue:
            result.log.append("检测到电源相关根因，检查同电源段 ECU 通讯状态")
            
            structured_data = state.get("structured_data", {})
            ecu_states = structured_data.get("ecu_states", [])
            dtc_codes = structured_data.get("dtc_codes", [])
            
            power_segment = self._extract_power_segment(root_cause)
            
            if power_segment:
                related_comm_errors = [
                    dtc for dtc in dtc_codes
                    if any(kw in dtc.get("description", "").lower() for kw in communication_keywords)
                ]
                
                if not related_comm_errors:
                    result.passed = False
                    result.conflict = f"电源段 {power_segment} 有断电问题但无对应通讯丢失 DTC"
                    result.log.append(f"警告: 电源段 {power_segment} 断电但无通讯丢失报错")
        
        return result
    
    def _check_dtc_coverage(self, state: DiagnosticState) -> AuditCheckResult:
        result = AuditCheckResult(
            check_name="dtc_coverage",
            passed=True,
            log=[],
        )
        
        structured_data = state.get("structured_data", {})
        dtc_codes = structured_data.get("dtc_codes", [])
        
        active_dtcs = [dtc for dtc in dtc_codes if dtc.get("status") == "Active"]
        
        if not active_dtcs:
            result.log.append("无 Active 状态的 DTC，跳过覆盖率检查")
            return result
        
        analysis_steps = state.get("analysis_steps", [])
        root_cause = state.get("preliminary_root_cause", "")
        
        covered_dtcs = []
        for dtc in active_dtcs:
            dtc_code = dtc.get("code", "")
            
            if dtc_code in root_cause or any(dtc_code in step for step in analysis_steps):
                covered_dtcs.append(dtc_code)
        
        uncovered = set(dtc.get("code") for dtc in active_dtcs) - set(covered_dtcs)
        coverage_ratio = len(covered_dtcs) / len(active_dtcs) if active_dtcs else 1.0
        min_ratio = float(self._config.get("dtc_coverage_min_ratio", 0.0))
        
        if uncovered and min_ratio > 0 and coverage_ratio < min_ratio:
            result.passed = False
            result.conflict = f"以下 Active DTC 未被诊断结果覆盖: {', '.join(uncovered)}"
            result.log.append(f"错误: Active DTC 覆盖率不足: {len(covered_dtcs)}/{len(active_dtcs)}")
        
        result.log.append(f"信息: Active DTC 覆盖率 {len(covered_dtcs)}/{len(active_dtcs)}")
        
        return result
    
    def _check_topology_consistency(self, state: DiagnosticState) -> AuditCheckResult:
        result = AuditCheckResult(
            check_name="topology_consistency",
            passed=True,
            log=[],
        )
        
        topology_graph = state.get("topology_graph")
        
        if not topology_graph:
            result.log.append("警告: 拓扑图为空，跳过拓扑一致性检查")
            return result
        
        root_cause = state.get("preliminary_root_cause", "")
        
        if not root_cause:
            result.passed = False
            result.conflict = "根因为空，跳过拓扑检查"
            result.log.append("错误: 根因为空，无法进行拓扑一致性检查")
            return result
        
        mentioned_ecus = self._extract_mentioned_ecus(root_cause)
        
        if not mentioned_ecus:
            result.log.append("信息: 根因中未提及具体 ECU，跳过拓扑一致性检查")
            return result
        
        topology_nodes = topology_graph.get("nodes", [])
        topology_ecu_ids = {str(node.get("ecu_id", "")).upper() for node in topology_nodes if node.get("ecu_id")}
        
        mentioned_in_topology = mentioned_ecus & topology_ecu_ids
        
        if mentioned_ecus and not mentioned_in_topology:
            result.passed = False
            result.conflict = f"根因中提到的 ECU {mentioned_ecus} 不在拓扑图中"
            result.log.append(f"错误: ECU 不在拓扑图中: {mentioned_ecus - topology_ecu_ids}")
        else:
            result.log.append(f"信息: 根因中的 ECU 都在拓扑图中: {mentioned_in_topology}")
        
        return result
    
    def _check_evidence_quality(self, state: DiagnosticState) -> AuditCheckResult:
        result = AuditCheckResult(
            check_name="evidence_quality",
            passed=True,
            log=[],
        )
        
        analysis_steps = state.get("analysis_steps", [])
        
        # CHAIN-005: 如果知识检索失败，降低证据质量要求
        risk_flags = state.get("risk_flags", [])
        base_min_steps = int(self._config.get("min_analysis_steps", 1))
        if "knowledge_retrieval_failed" in risk_flags:
            result.log.append("信息: 知识检索失败，降低证据质量要求")
            min_steps = max(1, base_min_steps)
        else:
            min_steps = base_min_steps
        
        if len(analysis_steps) < min_steps:
            result.passed = False
            result.conflict = "推理步骤不足，可能缺乏充分分析"
            result.log.append(f"警告: 推理步骤仅 {len(analysis_steps)} 步，可能不够充分")
            return result
        
        has_reasoning = any(
            "因为" in step or "由于" in step or "分析" in step or "reason" in step.lower()
            for step in analysis_steps
        )
        
        if not has_reasoning:
            result.passed = False
            result.conflict = "推理步骤缺乏逻辑推导过程"
            result.log.append("警告: 推理步骤缺乏明确的逻辑推导")
        
        return result
    
    def _check_confidence_threshold(self, state: DiagnosticState) -> AuditCheckResult:
        result = AuditCheckResult(
            check_name="confidence_threshold",
            passed=True,
            log=[],
        )
        
        confidence = state.get("confidence_score", 0.0)
        min_confidence = 0.75
        
        if confidence < min_confidence:
            result.passed = False
            result.conflict = f"置信度 {confidence:.2f} 低于阈值 {min_confidence}"
            result.severity = "warning"
            result.log.append(f"警告: 置信度低于阈值: {confidence:.2f} < {min_confidence}")
        else:
            result.log.append(f"信息: 置信度符合要求: {confidence:.2f}")
        
        return result

    def _check_root_cause_closure(self, state: DiagnosticState) -> AuditCheckResult:
        result = AuditCheckResult(
            check_name="root_cause_closure",
            passed=True,
            log=[],
        )
        root_cause = str(state.get("preliminary_root_cause", "") or "")
        steps_text = "\n".join(state.get("analysis_steps", []))
        focused_dtcs = list(state.get("focused_dtcs", []))
        if not root_cause:
            result.passed = False
            result.conflict = "根因为空，无法形成证据闭环"
            result.log.append("错误: 根因为空")
            return result
        has_query_success = "query_dtc" in steps_text and "success': True" in steps_text
        has_query_failure = "query_dtc" in steps_text and "success': False" in steps_text
        if not has_query_success:
            result.passed = False
            result.conflict = "缺少成功的DTC查询证据"
            result.log.append("错误: 未检测到成功DTC查询")
            return result
        mentioned_ecus = self._extract_mentioned_ecus(root_cause)
        if mentioned_ecus and focused_dtcs:
            focused_ecus = {str(item.get("ecu_id", "")).upper() for item in focused_dtcs if str(item.get("ecu_id", "")).strip()}
            if not (mentioned_ecus & focused_ecus):
                result.passed = False
                result.conflict = "根因ECU与聚焦DTC ECU不一致"
                result.log.append("错误: 根因与DTC证据链未对齐")
                return result
        if has_query_failure and "排除" not in steps_text and "exclude" not in steps_text.lower():
            result.passed = False
            result.conflict = "存在查询失败但缺少排除链路说明"
            result.log.append("错误: 缺少排除链路")
            return result
        result.log.append("信息: 根因与证据链闭环检查通过")
        return result
    
    def _extract_power_segment(self, text: str) -> str | None:
        # Updated regex to handle Chinese characters correctly
        match = re.search(r"(电源段?|Power.?Segment)[\s:]*([A-Za-z0-9_\u4e00-\u9fa5]+)", text, re.I)
        return match.group(2) if match else None
    
    def _extract_mentioned_ecus(self, text: str) -> set[str]:
        patterns = [
            re.compile(r"ECU[_\s(（:：-]*([0-9A-F]{4})", re.I),
            re.compile(r"\b([0-9A-F]{4})\b", re.I),
        ]
        ecus = set()
        for pattern in patterns:
            for item in pattern.findall(text):
                value = item.upper()
                if re.fullmatch(r"[0-9A-F]{4}", value):
                    ecus.add(value)
        return ecus
