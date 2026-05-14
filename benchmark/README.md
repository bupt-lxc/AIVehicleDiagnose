# Benchmark System

## 目标

标准化评测诊断链路，输出以下指标：

- 根因正确率
- Top-K 命中率
- 误诊率
- HITL 触发率
- 平均推理轮数
- 工具调用成本
- P95 延迟
- 审计拦截率

## 数据集格式

使用 `benchmark/cases.json`：

- `case_id`: 用例唯一标识
- `xml_path`: 电检报告路径
- `user_context`: 诊断上下文
- `enabled`: 是否纳入当前评测
- `label_status`: `labeled` 或 `ground_truth_pending`
- `expected_root_cause_keywords`: 根因正确率匹配关键词
- `expected_topk_keywords`: Top-K 命中候选关键词组
- `expected_hitl`: 预期是否触发 HITL

## 指标口径

- 根因正确率: `root_cause_correct == true` / `labeled_case_count`
- Top-K 命中率: `top_k_hit == true` / `labeled_case_count`
- 误诊率: `misdiagnosed == true` / `labeled_case_count`
- HITL 触发率: 仅统计诊断阶段 HITL，排除 `report_approval`
- 平均推理轮数: `mean(rca_iterations)`
- 工具调用成本: 对 `trace.tool_call` 按工具权重求和后取均值
- P95 延迟: 基于 `finished_at - started_at`
- 审计拦截率: `audit_passed == false` / `executed_case_count`

## 运行方式

```bash
python benchmark/run_benchmark.py
```

## 输出

运行后会在 `benchmark/results/<timestamp>/` 下生成：

- `benchmark_summary.json`
- `benchmark_report.md`
- `<case_id>/benchmark_case_result.json`
- `<case_id>/*_real_diagnostic_report.json`
- `<case_id>/*_full_chain_trace.json`
- `<case_id>/*_frontend_visible_record.json`

## 当前状态

- 已接入 `test4.xml` 作为已标注样例
- `test1.xml`、`test2.xml`、`test3.xml` 保留为 `ground_truth_pending`
- 后续只需补充真值关键词并启用 `enabled=true` 即可扩展为批量 benchmark
