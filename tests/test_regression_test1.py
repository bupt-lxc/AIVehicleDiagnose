import pytest
import asyncio
import os
import json
from pathlib import Path

# Add project root to sys.path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_real_flow import run_real_diagnostic

def test_automatic_parking_regression_test1():
    """
    关键回归用例：验证 test1.xml (自动泊车功能异常) 能否被正确解析并输出包含根因的最终报告。
    """
    # 运行真实诊断流程 (此函数已配置为读取 test1.xml 并输出到 logs/real_diagnostic_report.json)
    asyncio.run(run_real_diagnostic())
    
    report_path = Path("logs/real_diagnostic_report.json")
    assert report_path.exists(), "诊断报告文件未生成"
    
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    # 验证解析是否成功
    assert report.get("parse_success") is True, "XML 解析失败"
    
    # 验证结构化数据是否包含 DTC
    structured_data = report.get("structured_data", {})
    assert len(structured_data.get("dtc_codes", [])) > 0, "未能提取出 DTC 代码"
    
    # 验证是否成功生成推理步骤
    analysis_steps = report.get("analysis_steps", [])
    assert len(analysis_steps) > 0, "分析步骤为空，RCA 推理失败"
    
    # 验证根因是否被正确提取
    root_cause = report.get("preliminary_root_cause")
    assert root_cause is not None, "未提取出初步根因"
    assert "停车辅助" in root_cause or "通信" in root_cause or "ECU" in root_cause, "根因描述不符合预期"
    
    # 验证置信度
    confidence = report.get("confidence_score", 0.0)
    assert confidence > 0.0, "置信度应大于 0"

    print("✅ test1.xml 关键回归测试通过")
