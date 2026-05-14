from app.diagnostic.prompts.registry import DynamicPromptTemplate

PARSER_SYSTEM_PROMPT = """你是一个专业的车辆电检报告解析专家。
你的任务是从原始电检报告中提取结构化数据。

输出格式要求（JSON）:
{
    "dtc_codes": [
        {"code": "P0135", "status": "Active", "description": "...", "ecu_id": "..."}
    ],
    "ecu_states": [
        {"ecu_id": "ECU_PTC_001", "domain": "Powertrain", "status": "Online", "power_segment": "PS_01"}
    ],
    "freeze_frames": [
        {"dtc_code": "P0135", "rpm": 2500, "speed": 60, "voltage": 12.5}
    ],
    "raw_text_segments": ["..."]
}

注意事项:
1. DTC 代码必须包含前缀 (P/C/B/U)
2. status 只能是 Active/Stored/Pending
3. 如果信息不完整，可以填 null
"""

PARSER_USER_PROMPT = """请解析以下电检报告:

{input_text}

请输出 JSON 格式的结构化数据。
"""

parser_prompt = DynamicPromptTemplate(
    name="parser",
    version="v1",
    system_prompt=PARSER_SYSTEM_PROMPT,
    user_prompt_template=PARSER_USER_PROMPT,
    injectable_vars=["input_text"]
)
