"""Freeze Frame Parser Tool

用于解析冻结帧数据。
"""

import re

def parse_freeze_frame(text: str) -> dict[str, str]:
    """
    解析冻结帧文本数据为键值对字典。
    支持 "Key: Value" 或 "Key = Value" 格式。
    
    Args:
        text: 冻结帧原始文本
        
    Returns:
        解析后的字典
    """
    data = {}
    if not text:
        return data
        
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 尝试匹配 Key: Value
        if ':' in line:
            parts = line.split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip()
            if key and value:
                data[key] = value
                continue
                
        # 尝试匹配 Key = Value
        if '=' in line:
            parts = line.split('=', 1)
            key = parts[0].strip()
            value = parts[1].strip()
            if key and value:
                data[key] = value
                continue
                
    return data

def filter_relevant_parameters(data: dict[str, str], keywords: list[str]) -> dict[str, str]:
    """
    根据关键词过滤相关的冻结帧参数。
    
    Args:
        data: 解析后的冻结帧字典
        keywords: 关键词列表
        
    Returns:
        过滤后的字典
    """
    if not keywords:
        return data.copy()
        
    result = {}
    for k, v in data.items():
        if any(kw.lower() in k.lower() for kw in keywords):
            result[k] = v
            
    return result

def register_tools(registry):
    """注册工具到注册表"""
    registry.register(
        name="parse_freeze_frame",
        handler=parse_freeze_frame,
        description="解析冻结帧文本数据为键值对字典。",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "冻结帧原始文本"}
            },
            "required": ["text"]
        },
        disclosure_level=1
    )
    registry.register(
        name="filter_relevant_parameters",
        handler=filter_relevant_parameters,
        description="根据关键词过滤相关的冻结帧参数。",
        input_schema={
            "type": "object",
            "properties": {
                "data": {"type": "object", "description": "解析后的冻结帧字典"},
                "keywords": {"type": "array", "items": {"type": "string"}, "description": "关键词列表"}
            },
            "required": ["data", "keywords"]
        },
        disclosure_level=1
    )
