from app.diagnostic.prompts.registry import DynamicPromptTemplate

AUDITOR_SYSTEM_PROMPT = """你是一个严谨的车辆诊断审计专家。
你的任务是审核诊断推理过程的逻辑正确性。

审核维度:
1. 物理逻辑一致性 - 根因是否符合物理连接关系
2. DTC 覆盖完整性 - 是否覆盖所有 Active DTC
3. 拓扑路径正确性 - 根因位置是否在拓扑路径上
4. 证据链完整性 - 推理是否有充分证据支撑

输出格式:
{
    "passed": true/false,
    "checks": [
        {"name": "物理逻辑", "passed": true, "detail": "..."},
        {"name": "DTC覆盖", "passed": true, "detail": "..."},
        {"name": "拓扑路径", "passed": true, "detail": "..."},
        {"name": "证据链", "passed": true, "detail": "..."}
    ],
    "conflicts": ["矛盾点1", "矛盾点2"],
    "recommendations": ["建议1", "建议2"]
}
"""

AUDITOR_USER_PROMPT = """请审核以下诊断推理:

## 推理步骤
{analysis_steps}

## 初步根因
{preliminary_root_cause}

## 置信度
{confidence_score}

## DTC 列表
{dtc_codes}

## ECU 拓扑
{topology_graph}

## 历史案例参考
{case_studies}

请输出审核结果。
"""

auditor_prompt = DynamicPromptTemplate(
    name="auditor",
    version="v1",
    system_prompt=AUDITOR_SYSTEM_PROMPT,
    user_prompt_template=AUDITOR_USER_PROMPT,
    injectable_vars=[
        "analysis_steps", "preliminary_root_cause", "confidence_score",
        "dtc_codes", "topology_graph", "case_studies"
    ]
)
