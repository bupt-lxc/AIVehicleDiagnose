from __future__ import annotations

from typing import TYPE_CHECKING
import json
import re

from loguru import logger

from app.diagnostic.validators import InputValidator, StructuredDataValidator
from app.diagnostic.state import DiagnosticState
from app.diagnostic.prompts.registry import PromptRegistry
from app.diagnostic.tools.hex_decoder import HexDecoder
from app.diagnostic.tools.report_cleaner import ReportCleaner
from app.diagnostic.prioritizer import DTCPrioritizer, DTCSeverity

if TYPE_CHECKING:
    from app.providers import LLMProvider

logger = logger.bind(component="ParserAgent")


class ParseError(Exception):
    """解析错误异常"""
    
    def __init__(self, message: str, recoverable: bool = True):
        self.message = message
        self.recoverable = recoverable
        super().__init__(message)


class ParserAgent:
    NODE_NAME = "parser"
    
    MAX_INPUT_LENGTH = 100000
    MAX_RETRIES = 3
    LOCAL_LLM_MAX_RETRIES = 2
    
    def __init__(
        self,
        local_llm: "LLMProvider | None" = None,
        cloud_llm: "LLMProvider | None" = None,
        prompt_registry: PromptRegistry | None = None,
        config: dict | None = None,
    ):
        self._local_llm = local_llm
        self._cloud_llm = cloud_llm
        self._prompt_registry = prompt_registry or PromptRegistry()
        self._hex_decoder = HexDecoder()
        self._report_cleaner = ReportCleaner()
        self._input_validator = InputValidator()
        self._data_validator = StructuredDataValidator()
        self._prioritizer = DTCPrioritizer()
        self._config = config or {}
        
        logger.info("[ParserAgent] 初始化完成")
    
    def parse(self, state: DiagnosticState) -> dict:
        raw_input = state.get("raw_input", "")
        
        validation_result = self._input_validator.validate_raw_input(raw_input)
        
        if not validation_result.is_valid:
            logger.error("[ParserAgent] 输入验证失败: {}", validation_result.errors)
            return {
                "parse_success": False,
                "parse_error": "; ".join(validation_result.errors),
                "exit_code": 1,  # Invalid Input
                "current_node": self.NODE_NAME,
                "all_dtcs": [],
                "focused_dtcs": [],
                "deferred_dtcs": [],
                "dtc_selection_log": [],
            }
        
        if validation_result.warnings:
            logger.warning("[ParserAgent] 输入警告: {}", validation_result.warnings)
        
        cleaned_input = validation_result.sanitized_data
        
        try:
            cleaned_input = self._report_cleaner.clean(cleaned_input)
            cleaned_input = self._hex_decoder.decode(cleaned_input)
        except Exception as e:
            logger.error("[ParserAgent] 预处理失败: {}", e)
            return {
                "parse_success": False,
                "parse_error": f"预处理失败: {str(e)}",
                "exit_code": 2,  # Parser Failure
                "current_node": self.NODE_NAME,
                "all_dtcs": [],
                "focused_dtcs": [],
                "deferred_dtcs": [],
                "dtc_selection_log": [],
            }
        
        retry_state = state.get("retry_state", {})
        parser_retries = retry_state.get("parser_retries", 0)
        
        use_local = parser_retries < self.LOCAL_LLM_MAX_RETRIES
        
        try:
            structured_data = self._parse_with_llm(cleaned_input, use_local)
            
            # 优先级评定与过滤
            user_context = state.get("user_context", "")
            dtcs = structured_data.get("dtc_codes", [])
            
            # 获取过滤阈值 (默认 LOW)
            threshold_str = self._config.get("dtc_filter_threshold", "Low")
            try:
                threshold = DTCSeverity(threshold_str)
            except ValueError:
                threshold = DTCSeverity.LOW
                
            processed_dtcs = self._prioritizer.process_dtcs(dtcs, user_context, threshold)
            structured_data["dtc_codes"] = processed_dtcs
            
            logger.info("[ParserAgent] DTC处理完成: 原始 {} -> 过滤后 {}", len(dtcs), len(processed_dtcs))
            
            data_validation = self._data_validator.validate(structured_data)
            
            if not data_validation.is_valid:
                logger.warning("[ParserAgent] 数据验证警告: {}", data_validation.warnings)
                
                if data_validation.errors:
                    logger.error("[ParserAgent] 数据验证失败: {}", data_validation.errors)
                    return self._handle_parse_failure(
                        state,
                        f"数据验证失败: {'; '.join(data_validation.errors)}",
                        parser_retries,
                    )
            
            logger.info("[ParserAgent] 解析成功: {} 个 DTC", 
                       len(structured_data.get("dtc_codes", [])))
            
            return {
                "structured_data": structured_data,
                "parse_success": True,
                "parse_error": None,
                "all_dtcs": list(processed_dtcs),
                "focused_dtcs": list(processed_dtcs),
                "deferred_dtcs": [],
                "dtc_selection_log": [
                    {
                        "dtc_code": item.get("code"),
                        "selected": True,
                        "reason": "parser_initial_focus",
                        "priority_score": item.get("priority_score"),
                        "severity": item.get("severity"),
                    }
                    for item in processed_dtcs
                ],
                "retry_state": {**state.get("retry_state", {}), "parser_retries": 0},
                "current_node": self.NODE_NAME,
            }
            
        except ParseError as e:
            logger.error("[ParserAgent] 解析错误: {}", e.message)
            return self._handle_parse_failure(state, e.message, parser_retries)
        
        except Exception as e:
            logger.exception("[ParserAgent] 未知错误: {}", e)
            return self._handle_parse_failure(state, f"未知错误: {str(e)}", parser_retries)
    
    def _sanitize_for_prompt(self, text: str) -> str:
        # Remove potential prompt injection patterns
        sanitized = re.sub(r"(ignore|system|instruction|previous).*:", "", text, flags=re.I)
        # Escape special characters
        sanitized = sanitized.replace("{", "{{").replace("}", "}}")
        # Limit length
        return sanitized[:50000]

    def _parse_with_llm(
        self,
        cleaned_input: str,
        use_local: bool,
    ) -> dict:
        llm = self._local_llm if use_local else self._cloud_llm
        
        if llm is None:
            fallback_llm = self._cloud_llm if use_local else self._local_llm
            if fallback_llm is None:
                raise ParseError("无可用的 LLM", recoverable=False)
            llm = fallback_llm
            logger.info("[ParserAgent] 切换到 {} LLM", "云端" if llm == self._cloud_llm else "本地")
        
        sanitized_input = self._sanitize_for_prompt(cleaned_input)
        
        try:
            system_prompt = self._prompt_registry.get_system_prompt("parser")
            prompt = self._prompt_registry.get("parser", input_text=sanitized_input)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        except KeyError:
            # Fallback if prompt not found in registry (e.g. during tests without registry setup)
            prompt = self._build_prompt(sanitized_input)
            messages = [{"role": "user", "content": prompt}]

        response = llm.generate(
            messages=messages,
            temperature=0.1,
        )
        
        return self._parse_llm_response(response.content)
    
    def _build_prompt(self, input_text: str) -> str:
        return f"""请解析以下电检报告，输出 JSON 格式:

{input_text}

输出格式:
{{
    "dtc_codes": [
        {{"code": "P0135", "status": "Active", "description": "..."}}
    ],
    "ecu_states": [
        {{"ecu_id": "...", "domain": "Powertrain", "status": "Online"}}
    ],
    "freeze_frames": [],
    "raw_text_segments": []
}}

注意:
1. DTC 代码格式: P/B/C/U + 4位数字 (如 P0135)
2. status 只能是 Active/Stored/Pending
3. 如果无法解析，返回空数组
"""
    
    def _parse_llm_response(self, response: str) -> dict:
        json_match = re.search(r"\{[\s\S]*\}", response)
        
        if not json_match:
            raise ParseError("LLM 响应中未找到有效 JSON", recoverable=True)
        
        json_str = json_match.group()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # Try to fix common issues like trailing commas
                fixed_json = re.sub(r",\s*([\]}])", r"\1", json_str)
                data = json.loads(fixed_json)
            except json.JSONDecodeError as e:
                raise ParseError(f"JSON 解析失败: {str(e)}", recoverable=True)
            
        if not isinstance(data, dict):
            raise ParseError("JSON 根元素必须是对象", recoverable=True)
        
        data.setdefault("dtc_codes", [])
        data.setdefault("ecu_states", [])
        data.setdefault("freeze_frames", [])
        data.setdefault("raw_text_segments", [])
        
        # EDGE-002: 移除硬性数量限制，允许处理大量DTC
        # 之前限制为100是不合理的，实际场景可能更多
        dtc_codes = data.get("dtc_codes", [])
        if len(dtc_codes) > 1000:
            logger.warning(f"DTC数量极大({len(dtc_codes)})，仅记录日志警告")
        
        return data
    
    def _handle_parse_failure(
        self,
        state: DiagnosticState,
        error_message: str,
        parser_retries: int,
    ) -> dict:
        retry_state = state.get("retry_state", {})
        retry_state["parser_retries"] = parser_retries + 1
        
        if parser_retries + 1 >= self.MAX_RETRIES:
            logger.warning("[ParserAgent] 达到最大重试次数 ({})，返回失败状态", self.MAX_RETRIES)
            return {
                "parse_success": False,
                "parse_error": error_message,
                "retry_state": retry_state,
                "exit_code": 2,  # Parser Failure
                "current_node": self.NODE_NAME,
                "all_dtcs": [],
                "focused_dtcs": [],
                "deferred_dtcs": [],
                "dtc_selection_log": [],
            }
        
        logger.info("[ParserAgent] 将在下一轮重试 (尝试 {}/{})", 
                   parser_retries + 1, self.MAX_RETRIES)
        
        return {
            "parse_success": False,
            "parse_error": error_message,
            "retry_state": retry_state,
            "current_node": self.NODE_NAME,
            "all_dtcs": [],
            "focused_dtcs": [],
            "deferred_dtcs": [],
            "dtc_selection_log": [],
        }
