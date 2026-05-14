# app/diagnostic/validators.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import re


@dataclass(slots=True)
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    sanitized_data: Any | None


class InputValidator:
    RAW_INPUT_MAX_LENGTH = 100000
    RAW_INPUT_MIN_LENGTH = 10
    DTC_PATTERN = re.compile(r"^[PBCU][0-9]{4}$")
    ECU_ID_PATTERNS = [
        re.compile(r"^ECU_[A-Z]{3}_\d{3}$", re.I),
        re.compile(r"^ECU_[A-Z]+_\d+$", re.I),
        re.compile(r"^[A-Z]{2,5}[_\-]?\d{2,4}$", re.I),
        re.compile(r"^ECU[A-Z]*\d*$", re.I),
    ]
    
    def validate_raw_input(self, raw_input: str) -> ValidationResult:
        errors = []
        warnings = []
        
        if not raw_input or not raw_input.strip():
            errors.append("输入不能为空")
            return ValidationResult(False, errors, warnings, None)
        
        if len(raw_input) < self.RAW_INPUT_MIN_LENGTH:
            errors.append(f"输入长度不足，最少需要 {self.RAW_INPUT_MIN_LENGTH} 字符")
        
        if len(raw_input) > self.RAW_INPUT_MAX_LENGTH:
            errors.append(f"输入长度超限，最多允许 {self.RAW_INPUT_MAX_LENGTH} 字符")
        
        sanitized = self._sanitize_input(raw_input)
        
        # EDGE-001: 确保清洗后的内容不为空
        if not sanitized.strip():
            errors.append("输入内容包含无效字符")
            return ValidationResult(False, errors, warnings, None)
        
        if not self._contains_valid_content(sanitized):
            warnings.append("输入内容可能不包含有效的诊断信息")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized,
        )
    
    def _sanitize_input(self, raw_input: str) -> str:
        sanitized = raw_input.strip()
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", sanitized)
        return sanitized
    
    def _contains_valid_content(self, text: str) -> bool:
        has_dtc = bool(self.DTC_PATTERN.search(text))
        has_hex = bool(re.search(r"[0-9A-Fa-f]{4,}", text))
        has_ecu = any(pattern.search(text) for pattern in self.ECU_ID_PATTERNS)
        has_error_desc = bool(re.search(r"(error|fault|fail|丢失|故障|异常)", text, re.I))
        
        return has_dtc or has_hex or has_ecu or has_error_desc


class StructuredDataValidator:
    def validate(self, data: dict) -> ValidationResult:
        errors = []
        warnings = []
        
        dtc_codes = data.get("dtc_codes", [])
        ecu_states = data.get("ecu_states", [])
        freeze_frames = data.get("freeze_frames", [])
        
        if not dtc_codes and not ecu_states and not freeze_frames:
            errors.append("结构化数据为空，无法进行诊断")
        
        for dtc in dtc_codes:
            if not self._validate_dtc(dtc):
                errors.append(f"无效的 DTC 格式: {dtc}")
        
        for ecu in ecu_states:
            if not self._validate_ecu_state(ecu):
                warnings.append(f"ECU 状态信息不完整: {ecu}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=data,
        )
    
    def _validate_dtc(self, dtc: dict) -> bool:
        if not isinstance(dtc, dict):
            return False
        code = dtc.get("code", "")
        if not InputValidator.DTC_PATTERN.match(code):
            return False
        if dtc.get("status", "").title() not in ["Active", "Stored", "Pending"]:
            return False
        return True
    
    def _validate_ecu_state(self, ecu: dict) -> bool:
        return bool(ecu.get("ecu_id") and ecu.get("status"))


class StateTransitionValidator:
    ALLOWED_TRANSITIONS = {
        "input_validator": ["parser", "error_handler"],
        "parser": ["hitl_parse_confirm", "parser", "parser_fallback", "error_handler"],
        "parser_fallback": ["hitl_parse_confirm", "error_handler"],
        "hitl_parse_confirm": ["surveyor"],
        "surveyor": ["mcp_librarian"],
        "mcp_librarian": ["rca_engine"],
        "rca_engine": ["hitl_low_confidence", "auditor"],
        "hitl_low_confidence": ["auditor", "rca_engine"],
        "auditor": ["hitl_audit_conflict", "hitl_report_approval"],
        "hitl_audit_conflict": ["rca_engine", "hitl_report_approval"],
        "hitl_report_approval": ["report_generator", "rca_engine", "hitl_low_confidence"],
        "report_generator": [],
        "error_handler": [],
    }
    
    def validate_transition(
        self,
        from_node: str,
        to_node: str,
        state: dict,
    ) -> ValidationResult:
        errors = []
        warnings = []
        
        if from_node not in self.ALLOWED_TRANSITIONS:
            errors.append(f"未知源节点: {from_node}")
        elif to_node not in self.ALLOWED_TRANSITIONS.get(from_node, []):
            errors.append(f"不允许的转换: {from_node} -> {to_node}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=None,
        )
