# app/diagnostic/state_checker.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class StateIntegrityReport:
    is_valid: bool
    missing_fields: list[str]
    invalid_fields: list[str]
    inconsistencies: list[str]


class StateIntegrityChecker:
    REQUIRED_FIELDS = [
        "raw_input",
        "structured_data", "parse_success", "current_node",
        "retry_state", "exit_code",
        "all_dtcs", "focused_dtcs", "deferred_dtcs", "dtc_selection_log",
        "confidence_history", "hypothesis_queue", "evidence_graph", "rca_iteration",
        "transition_log", "operator_trace", "degraded_mode", "knowledge_degraded",
        "thread_id", "run_id", "started_at",
    ]
    
    CONDITIONAL_REQUIREMENTS = {
        "parse_success": {
            True: ["structured_data.dtc_codes", "all_dtcs", "focused_dtcs", "deferred_dtcs", "dtc_selection_log"],
            False: ["parse_error"],
        },
        "audit_passed": {
            False: ["conflicts"],
        },
        "hitl_required": {
            True: ["current_hitl_point", "hitl_question", "hitl_options"],
        },
        "report_approved": {
            True: ["final_report"],
        },
    }
    
    def check(self, state: dict) -> StateIntegrityReport:
        missing_fields = []
        invalid_fields = []
        inconsistencies = []
        
        for field in self.REQUIRED_FIELDS:
            if field not in state:
                missing_fields.append(field)
        
        for field, conditions in self.CONDITIONAL_REQUIREMENTS.items():
            field_value = state.get(field)
            
            for condition_value, required_subfields in conditions.items():
                if field_value == condition_value:
                    for subfield in required_subfields:
                        if "." in subfield:
                            parts = subfield.split(".")
                            current = state.get(parts[0])
                            for part in parts[1:]:
                                if current is None:
                                    missing_fields.append(subfield)
                                    break
                                current = current.get(part) if isinstance(current, dict) else None
                        elif subfield not in state or state.get(subfield) is None:
                            missing_fields.append(subfield)
        
        if state.get("confidence_score", 0) < 0 or state.get("confidence_score", 0) > 1:
            invalid_fields.append("confidence_score")
        
        if state.get("exit_code") not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            invalid_fields.append("exit_code")
        
        if state.get("parse_success") and not state.get("structured_data", {}).get("dtc_codes"):
            inconsistencies.append("parse_success=True 但 dtc_codes 为空")
        if state.get("parse_success") and not isinstance(state.get("dtc_selection_log", []), list):
            inconsistencies.append("parse_success=True 但 dtc_selection_log 非list")
        if state.get("focused_dtcs") and not state.get("all_dtcs"):
            inconsistencies.append("focused_dtcs 非空但 all_dtcs 为空")
        
        return StateIntegrityReport(
            is_valid=len(missing_fields) == 0 and len(invalid_fields) == 0,
            missing_fields=missing_fields,
            invalid_fields=invalid_fields,
            inconsistencies=inconsistencies,
        )
