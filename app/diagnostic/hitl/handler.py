import logging
from datetime import datetime, timezone
from app.diagnostic.state import DiagnosticState, HITLFeedback

logger = logging.getLogger(__name__)

class HITLHandler:
    """Human-in-the-Loop 处理器，用于在关键节点生成人工介入请求并处理反馈。"""
    
    def handle_parse_confirm(self, state: DiagnosticState) -> DiagnosticState:
        """处理解析确认的 HITL 请求"""
        dtc_codes = state.get("structured_data", {}).get("dtc_codes", [])
        
        if len(dtc_codes) <= 1:
            priority_dtcs = [d["code"] for d in dtc_codes]
            return {**state, "hitl_required": False, "priority_dtcs": priority_dtcs}
        
        # LOGIC-005: 限制选项数量，避免 D 选项冲突
        max_options = 10
        count = len(dtc_codes)
        
        if count > max_options:
            options = [chr(65 + i) for i in range(max_options)] + ["Z"]
        else:
            options = [chr(65 + i) for i in range(count)] + ["Z"]
        
        return {
            **state,
            "hitl_required": True,
            "current_hitl_point": "parse_confirm",
            "hitl_question": f"检测到 {count} 个 DTC，请选择重点分析的 DTC (Z=全部)",
            "hitl_options": options,
        }
    
    def handle_low_confidence(self, state: DiagnosticState) -> DiagnosticState:
        """处理低置信度的 HITL 请求"""
        possible_causes = self._extract_possible_causes(state.get("analysis_steps", []))
        confidence = state.get("confidence_score", 0.0)
        
        options = [f"{chr(65 + i)}: {cause}" for i, cause in enumerate(possible_causes)]
        options.append(f"{chr(65 + len(possible_causes))}: 其他（手动描述）")
        
        return {
            **state,
            "hitl_required": True,
            "current_hitl_point": "low_confidence",
            "hitl_question": f"当前置信度 {confidence:.0%}，请选择最可能的原因",
            "hitl_options": options,
        }
    
    def handle_audit_conflict(self, state: DiagnosticState) -> DiagnosticState:
        """处理审计冲突的 HITL 请求"""
        conflicts = state.get("conflicts", [])
        
        # COND-002: 明确选项含义
        return {
            **state,
            "hitl_required": True,
            "current_hitl_point": "audit_conflict",
            "hitl_question": f"检测到 {len(conflicts)} 个逻辑矛盾:\n" + 
                            "\n".join(f"- {c}" for c in conflicts) +
                            "\n\n请选择处理方式",
            "hitl_options": ["A: 忽略冲突继续", "B: 重新分析", "C: 提供更多信息"],
        }
    
    def handle_report_approval(self, state: DiagnosticState) -> DiagnosticState:
        """处理报告审批的 HITL 请求"""
        return {
            **state,
            "hitl_required": True,
            "current_hitl_point": "report_approval",
            "hitl_question": "诊断报告已生成，请审批",
            "hitl_options": ["A", "B", "C"],
        }
    
    def process_feedback(
        self,
        state: DiagnosticState,
        user_choice: str,
        user_input: str | None = None,
    ) -> DiagnosticState:
        """处理用户的 HITL 反馈，并更新状态"""
        raw_options = state.get("hitl_options", [])
        normalized_options = []
        for option in raw_options:
            option_text = str(option)
            token = option_text.split(":", 1)[0].strip().upper()
            if token:
                normalized_options.append(token)
        normalized_choice = str(user_choice).strip().upper()
        if normalized_choice not in normalized_options:
            return {
                **state,
                "hitl_required": True,
                "hitl_question": f"无效选择 '{user_choice}'，请从 {raw_options} 中选择",
                "error_message": "invalid_choice",
            }
            
        feedback = HITLFeedback(
            point=state.get("current_hitl_point", "unknown"),
            question=state.get("hitl_question", ""),
            options=state.get("hitl_options", []),
            user_choice=normalized_choice,
            user_input=user_input,
            timestamp=datetime.now(timezone.utc).isoformat(),
            timed_out=False,
        )
        
        feedbacks = list(state.get("hitl_feedbacks", []))
        # STATE-002: 限制历史记录长度
        MAX_FEEDBACKS = 50  # 增加历史记录容量
        if len(feedbacks) >= MAX_FEEDBACKS:
            feedbacks = feedbacks[-MAX_FEEDBACKS:]
            
        feedbacks.append(feedback)
        
        new_state = {
            **state,
            "hitl_feedbacks": feedbacks,
            "hitl_required": False,
            "current_hitl_point": None,
        }
        
        current_point = state.get("current_hitl_point")
        
        # 根据不同 HITL 点处理用户选择
        if current_point == "parse_confirm":
            structured_data = state.get("structured_data", {})
            dtc_codes = structured_data.get("dtc_codes", [])
            all_dtcs = [d["code"] for d in dtc_codes]
            
            if normalized_choice == "Z":
                new_state["priority_dtcs"] = all_dtcs
            else:
                idx = ord(normalized_choice) - 65
                if 0 <= idx < len(all_dtcs):
                    new_state["priority_dtcs"] = [all_dtcs[idx]]
                else:
                    logger.warning(f"Invalid HITL choice index {idx} for {len(all_dtcs)} items. Defaulting to all.")
                    new_state["priority_dtcs"] = all_dtcs
                
        elif current_point == "low_confidence":
            possible_causes = self._extract_possible_causes(state.get("analysis_steps", []))
            other_letter = chr(65 + len(possible_causes))
            if normalized_choice == other_letter:
                new_state["user_description"] = user_input
                if user_input:
                    new_state["user_context"] = f"{state.get('user_context', '')} {user_input}".strip()
            else:
                idx = ord(normalized_choice) - 65
                if 0 <= idx < len(possible_causes):
                    new_state["human_hint"] = possible_causes[idx]
        elif current_point in {"focus_adjustment", "knowledge_retrieval"}:
            if user_input:
                tokens = [item.strip().upper() for item in user_input.replace(",", " ").split() if len(item.strip()) >= 2]
                if tokens:
                    focused = [d for d in state.get("all_dtcs", []) if str(d.get("ecu_id", "")).upper() in tokens]
                    if focused:
                        new_state["focused_dtcs"] = focused
                        new_state["deferred_dtcs"] = [d for d in state.get("all_dtcs", []) if d not in focused]
                        structured = dict(state.get("structured_data", {}))
                        structured["dtc_codes"] = focused
                        new_state["structured_data"] = structured
                        new_state["operator_trace"] = list(state.get("operator_trace", [])) + [
                            {
                                "type": "hitl_focus_override",
                                "preferred_ecus": tokens,
                                "focused_count": len(focused),
                            }
                        ]
                    
        elif current_point == "audit_conflict":
            if normalized_choice.startswith("A"):
                new_state["conflict_ignored"] = True
                new_state["risk_flags"] = state.get("risk_flags", []) + ["audit_conflict_ignored"]
                
        elif current_point == "report_approval":
            if normalized_choice == "A":
                new_state["report_approved"] = True
                
        return new_state

    def _extract_possible_causes(self, analysis_steps: list[str]) -> list[str]:
        """从分析步骤中提取可能的原因列表"""
        causes = []
        for step in analysis_steps:
            # 简单的启发式规则：查找包含“可能原因”等关键词的步骤
            if "可能原因" in step or "possible cause" in step.lower() or "root cause" in step.lower():
                causes.append(step.strip())
                
        # 如果未找到，提供一些默认的通用原因选项
        if not causes:
            causes = [
                "通信网络故障 (CAN/LIN 等总线异常)",
                "供电系统故障 (电池/保险丝/接地异常)",
                "ECU 硬件/软件故障",
                "传感器或执行器故障"
            ]
            
        return causes[:4] # 限制最多4个选项，以便匹配A/B/C/D
