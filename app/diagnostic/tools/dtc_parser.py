"""DTC Parser Tool

用于解析和验证 DTC 故障码格式。
"""

import re

def extract_dtc_codes(text: str) -> list[str]:
    """
    从文本中提取符合 OBD-II 标准格式 (P/C/B/U + 4位数字) 的 DTC 代码。
    
    Args:
        text: 输入文本
        
    Returns:
        提取到的 DTC 代码列表 (去重且大写)
    """
    if not text:
        return []
        
    # 匹配 P/C/B/U 开头，后跟4位十六进制数字 (部分厂商可能是十六进制)
    # 这里严格遵循标准: [PCBU]\d{4}，如果需要支持 hex，可以改为 [PCBU][0-9A-F]{4}
    pattern = r'\b([PCBU][0-9A-F]{4})\b'
    matches = re.findall(pattern, text.upper())
    
    return sorted(list(set(matches)))

def validate_dtc_format(code: str) -> bool:
    """
    验证单个 DTC 代码格式是否合法。
    
    Args:
        code: DTC 代码
        
    Returns:
        是否合法
    """
    if not code:
        return False
        
    return bool(re.match(r'^[PCBU][0-9A-F]{4}$', code.upper()))

def register_tools(registry):
    """注册工具到注册表"""
    registry.register(
        name="extract_dtc_codes",
        handler=extract_dtc_codes,
        description="从文本中提取符合 OBD-II 标准格式 (P/C/B/U + 4位数字) 的 DTC 代码。",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "输入文本"}
            },
            "required": ["text"]
        },
        disclosure_level=0
    )
    registry.register(
        name="validate_dtc_format",
        handler=validate_dtc_format,
        description="验证单个 DTC 代码格式是否合法。",
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "DTC 代码"}
            },
            "required": ["code"]
        },
        disclosure_level=0
    )
