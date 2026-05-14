"""
HITL Prompts for AutoDiagnose Multi-Agent System.

This module defines the prompt templates used when human intervention is required.
"""

HITL_PARSE_CONFIRM = """
检测到 {count} 个 DTC 代码:
{dtc_list}

请选择优先分析的 DTC:
{options}

输入选项字母 (A/B/C/...) 或 D (全部分析):
"""

HITL_LOW_CONFIDENCE = """
当前诊断置信度: {confidence:.0%}

系统识别的可能原因:
{possible_causes}

请选择最可能的原因:
{options}

输入选项字母 (A/B/C/...) 或 D (其他 - 请描述):
"""

HITL_AUDIT_CONFLICT = """
检测到以下逻辑矛盾:
{conflicts}

请选择处理方式:
A) 忽略矛盾，继续生成报告
B) 返回重新分析
C) 提供额外信息后重新分析

输入选项字母:
"""

HITL_REPORT_APPROVAL = """
诊断报告已生成，请审批:

{report_preview}

请选择:
A) 批准报告
B) 需要修改
C) 需要更多信息

输入选项字母:
"""
