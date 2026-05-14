from enum import Enum
from typing import List, Dict, Any
import re
from loguru import logger

class DTCSeverity(str, Enum):
    CRITICAL = "Critical"  # 严重：影响行驶安全或主要功能
    HIGH = "High"          # 高：影响重要功能或排放
    MEDIUM = "Medium"      # 中：影响舒适性或辅助功能
    LOW = "Low"            # 低：信息性或轻微故障
    UNKNOWN = "Unknown"    # 未知

class DTCPrioritizer:
    """
    DTC 优先级评定与过滤组件
    负责根据规则自动评定 DTC 严重等级，计算优先级，并过滤无效 DTC
    """
    
    # 关键系统关键词
    CRITICAL_KEYWORDS = [
        "airbag", "brake", "abs", "esp", "steering", "engine", "transmission", 
        "battery", "bms", "hv", "motor", "inverter", "安全气囊", "刹车", "制动", 
        "转向", "发动机", "变速箱", "电池", "高压", "电机", "逆变器", "loss of communication"
    ]
    
    HIGH_KEYWORDS = [
        "emission", "sensor", "injector", "misfire", "catalyst", "排放", "传感器", 
        "喷油嘴", "缺火", "催化剂", "acc", "radar", "camera", "雷达", "摄像头"
    ]
    
    # DTC 前缀规则
    PREFIX_RULES = {
        "P": DTCSeverity.HIGH,     # 动力系统默认高
        "C": DTCSeverity.HIGH,     # 底盘系统默认高
        "B": DTCSeverity.MEDIUM,   # 车身系统默认中
        "U": DTCSeverity.MEDIUM,   # 网络通信默认中
    }

    def __init__(self):
        pass

    def _severity_to_value(self, severity: DTCSeverity | str) -> int:
        mapping = {
            DTCSeverity.CRITICAL: 4,
            DTCSeverity.HIGH: 3,
            DTCSeverity.MEDIUM: 2,
            DTCSeverity.LOW: 1,
            DTCSeverity.UNKNOWN: 0
        }
        if isinstance(severity, str):
            for key in mapping:
                if key.value == severity:
                    return mapping[key]
            return 0
        return mapping.get(severity, 0)

    def _value_to_severity(self, value: int) -> DTCSeverity:
        value = max(0, min(4, value))
        reverse = {
            4: DTCSeverity.CRITICAL,
            3: DTCSeverity.HIGH,
            2: DTCSeverity.MEDIUM,
            1: DTCSeverity.LOW,
            0: DTCSeverity.UNKNOWN
        }
        return reverse[value]

    def _metadata_severity(self, dtc: Dict[str, Any]) -> DTCSeverity:
        raw_data = dtc.get("raw_data", {}) or {}
        fault_event = str(raw_data.get("fault_event", "")).strip().lower()
        fault_status = str(raw_data.get("fault_status", "")).strip().lower()
        dtc_priority = str(raw_data.get("dtc_priority", "")).strip()

        try:
            priority_num = int(dtc_priority)
        except (ValueError, TypeError):
            priority_num = None

        if fault_event.startswith("ereignis") or fault_event == "event":
            if priority_num in (1, 2):
                severity = DTCSeverity.CRITICAL
            elif priority_num == 3:
                severity = DTCSeverity.HIGH
            elif priority_num in (4, 5):
                severity = DTCSeverity.MEDIUM
            elif priority_num == 6:
                severity = DTCSeverity.LOW
            else:
                severity = DTCSeverity.MEDIUM
        elif fault_event.startswith("information") or fault_event == "info":
            if priority_num in (1, 2):
                severity = DTCSeverity.HIGH
            elif priority_num == 3:
                severity = DTCSeverity.MEDIUM
            else:
                severity = DTCSeverity.LOW
        else:
            severity = DTCSeverity.UNKNOWN

        if fault_status.startswith("passiv") or str(dtc.get("status", "")).lower() in {"stored", "passive"}:
            downgraded = self._severity_to_value(severity) - 1
            return self._value_to_severity(downgraded)
        return severity

    def evaluate_severity(self, dtc_code: str, description: str = "") -> DTCSeverity:
        """
        根据 DTC 代码和描述评定严重等级
        """
        description = description.lower() if description else ""
        dtc_code = dtc_code.upper()
        
        # 1. 基于关键词的规则 (优先级最高)
        for keyword in self.CRITICAL_KEYWORDS:
            if keyword in description:
                return DTCSeverity.CRITICAL
                
        for keyword in self.HIGH_KEYWORDS:
            if keyword in description:
                return DTCSeverity.HIGH

        # 2. 基于 U 码特殊规则: U0 开头通常是全车网络问题，较为严重
        if dtc_code.startswith("U0"):
            return DTCSeverity.HIGH

        # 3. 基于前缀的默认规则
        prefix = dtc_code[0] if dtc_code else ""
        if prefix in self.PREFIX_RULES:
            return self.PREFIX_RULES[prefix]
            
        return DTCSeverity.UNKNOWN

    def calculate_priority(self, dtc: Dict[str, Any], user_context: str = "") -> int:
        """
        计算 DTC 优先级分数 (分数越高优先级越高)
        """
        base_score = 0
        severity = dtc.get("severity", DTCSeverity.UNKNOWN)
        
        # 基础分：基于严重等级
        if severity == DTCSeverity.CRITICAL:
            base_score += 100
        elif severity == DTCSeverity.HIGH:
            base_score += 70
        elif severity == DTCSeverity.MEDIUM:
            base_score += 40
        elif severity == DTCSeverity.LOW:
            base_score += 10
        
        # 状态加分：Active > Pending > Stored
        status = dtc.get("status", "").title()
        if status == "Active":
            base_score += 20
        elif status == "Pending":
            base_score += 10
        elif status == "Stored":
            base_score -= 5

        raw_data = dtc.get("raw_data", {}) or {}
        fault_event = str(raw_data.get("fault_event", "")).strip().lower()
        fault_status = str(raw_data.get("fault_status", "")).strip().lower()
        dtc_priority = str(raw_data.get("dtc_priority", "")).strip()

        if fault_event.startswith("ereignis") or fault_event == "event":
            base_score += 15
        elif fault_event.startswith("information") or fault_event == "info":
            base_score -= 10

        if fault_status.startswith("aktiv"):
            base_score += 10
        elif fault_status.startswith("passiv"):
            base_score -= 15

        try:
            priority_num = int(dtc_priority)
            if 1 <= priority_num <= 6:
                base_score += (7 - priority_num) * 5
        except (ValueError, TypeError):
            pass
            
        # 上下文相关性加分
        if user_context:
            user_context = user_context.lower()
            desc = (dtc.get("description") or "").lower()
            code = (dtc.get("code") or "").lower()
            ecu = (dtc.get("ecu_id") or "").lower()
            
            # 简单的关键词匹配
            context_keywords = [k.strip() for k in user_context.split() if len(k.strip()) > 1]
            for kw in context_keywords:
                if kw in desc or kw in code or kw in ecu:
                    base_score += 50  # 强相关，大幅加分
                    logger.info(f"DTC {code} 与用户描述 '{kw}' 相关，优先级提升")
                    break
        
        return base_score

    def process_dtcs(self, dtcs: List[Dict[str, Any]], user_context: str = "", filter_threshold: DTCSeverity = DTCSeverity.LOW) -> List[Dict[str, Any]]:
        """
        批量处理 DTC：评定等级 -> 计算优先级 -> 排序 -> 过滤
        """
        processed_dtcs = []
        
        # 定义严重等级的数值映射用于比较
        severity_value = {
            DTCSeverity.CRITICAL: 4,
            DTCSeverity.HIGH: 3,
            DTCSeverity.MEDIUM: 2,
            DTCSeverity.LOW: 1,
            DTCSeverity.UNKNOWN: 0
        }
        
        threshold_value = severity_value.get(filter_threshold, 0)
        
        for dtc in dtcs:
            inferred = self.evaluate_severity(dtc.get("code", ""), dtc.get("description", ""))
            metadata = self._metadata_severity(dtc)
            metadata_value = self._severity_to_value(metadata)
            if metadata_value > 0:
                final_value = metadata_value
            else:
                final_value = self._severity_to_value(inferred)
            dtc["severity"] = self._value_to_severity(final_value).value
            
            # 2. 过滤
            current_severity = dtc["severity"]
            # 注意：这里我们保留所有 >= 阈值的 DTC。UNKNOWN 也可能被保留，取决于策略，这里假设 UNKNOWN < LOW
            # 如果需要保留 UNKNOWN，可以特殊处理。
            # 暂时策略：只过滤明确低于阈值的。
            
            if self._severity_to_value(current_severity) < threshold_value:
                logger.debug(f"DTC {dtc.get('code')} (Severity: {current_severity}) 低于阈值 {filter_threshold}，已过滤")
                continue
                
            # 3. 计算优先级
            dtc["priority_score"] = self.calculate_priority(dtc, user_context)
            processed_dtcs.append(dtc)
            
        # 4. 排序：优先级分数降序
        processed_dtcs.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return processed_dtcs


class DTCFocusEngine:
    DEFAULT_MAX_FOCUSED = 12
    NETWORK_KEYWORDS = ("communication", "network", "bus", "can", "lin", "gateway", "通信", "总线", "网关")

    def __init__(self, max_focused: int = DEFAULT_MAX_FOCUSED):
        self._max_focused = max(3, int(max_focused))

    def focus(
        self,
        *,
        dtcs: List[Dict[str, Any]],
        user_context: str = "",
        preferred_ecus: list[str] | None = None,
    ) -> dict[str, Any]:
        preferred_ecus = [str(item).upper() for item in (preferred_ecus or []) if str(item).strip()]
        context = (user_context or "").lower()
        enriched: list[dict[str, Any]] = []
        for item in dtcs:
            dtc = dict(item)
            score = int(dtc.get("priority_score", 0) or 0)
            reasons: list[str] = []
            severity = str(dtc.get("severity", "Unknown"))
            if severity in {"Critical", "High"}:
                score += 30
                reasons.append("severity_priority")
            desc = str(dtc.get("description", "")).lower()
            code = str(dtc.get("code", "")).lower()
            ecu = str(dtc.get("ecu_id", "")).upper()
            if context and any(token and (token in desc or token in code or token in ecu.lower()) for token in re.split(r"\s+", context) if len(token) > 1):
                score += 40
                reasons.append("user_context_relevance")
            if code.startswith("u0") or any(key in desc for key in self.NETWORK_KEYWORDS):
                score += 20
                reasons.append("topology_risk")
            if preferred_ecus and ecu in preferred_ecus:
                score += 35
                reasons.append("hitl_preferred_ecu")
            dtc["focus_score"] = score
            dtc["focus_reasons"] = reasons or ["baseline_priority"]
            enriched.append(dtc)
        enriched.sort(key=lambda x: int(x.get("focus_score", 0)), reverse=True)
        focused = enriched[: self._max_focused]
        deferred = enriched[self._max_focused :]
        selection_log = [
            {
                "dtc_code": item.get("code"),
                "selected": idx < len(focused),
                "focus_score": item.get("focus_score", 0),
                "priority_score": item.get("priority_score"),
                "reasons": item.get("focus_reasons", []),
                "severity": item.get("severity"),
                "ecu_id": item.get("ecu_id"),
            }
            for idx, item in enumerate(enriched)
        ]
        return {
            "all_dtcs": enriched,
            "focused_dtcs": focused,
            "deferred_dtcs": deferred,
            "dtc_selection_log": selection_log,
        }
