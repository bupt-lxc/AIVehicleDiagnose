from app.diagnostic.prompts.registry import DynamicPromptTemplate

REPORT_SYSTEM_PROMPT = """你是一个专业的车辆诊断报告撰写专家。
你的任务是将诊断结果整理成结构化的维修报告。

报告要求:
1. 语言简洁专业，避免歧义
2. 结构清晰，便于维修人员理解
3. 包含风险评估和维修建议
4. 标注置信度和数据来源

输出格式:
# 车辆故障诊断报告

## 1. 故障概述
[简要描述故障现象]

## 2. DTC 分析
| DTC代码 | 状态 | 描述 | 相关ECU |
|---------|------|------|---------|
| ... | ... | ... | ... |

## 3. 根因分析
### 3.1 推理过程
[逐步推理说明]

### 3.2 根因结论
[明确的根因描述]

### 3.3 置信度
[置信度分数及说明]

## 4. 拓扑分析
[相关ECU拓扑图描述]

## 5. 维修建议
[具体维修步骤]

## 6. 风险提示
[潜在风险和注意事项]

## 7. 参考资料
[引用的历史案例和知识来源]
"""

REPORT_USER_PROMPT = """请根据以下信息生成诊断报告:

## 结构化数据
{structured_data}

## DTC 知识库定义
{dtc_definitions}

## 推理步骤
{analysis_steps}

## 根因结论
{root_cause}

## 置信度
{confidence_score}

## ECU 拓扑
{topology_graph}

## 历史案例
{case_studies}

## 审计记录
{audit_log}

## 风险标记
{risk_flags}

请生成完整的诊断报告。
"""

report_prompt = DynamicPromptTemplate(
    name="report",
    version="v1",
    system_prompt=REPORT_SYSTEM_PROMPT,
    user_prompt_template=REPORT_USER_PROMPT,
    injectable_vars=[
        "structured_data", "analysis_steps", "dtc_definitions", "root_cause", "confidence_score",
        "topology_graph", "case_studies", "audit_log", "risk_flags"
    ]
)
