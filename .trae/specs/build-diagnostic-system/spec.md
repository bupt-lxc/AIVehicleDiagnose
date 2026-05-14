# AutoDiagnose Multi-Agent System Spec

## Why
需要一个智能化故障诊断平台来解析车辆电检报告，结合车辆物理拓扑与技术知识库，通过多智能体协同输出具备逻辑严密性的根因分析（RCA）报告。该系统将作为现有RAG系统的上层应用模块。

## What Changes
- 新增 `diagnostic/` 模块，包含状态定义 (`state.py`)、图编排 (`graph.py`) 和配置 (`config.py`)。
- 新增多智能体实现 (`diagnostic/agents/`)：`Input Validator`, `Parser Agent`, `Surveyor Agent`, `MCP Librarian`, `RCA Engine`, `Auditor Agent`, `Report Generator`。
- 新增工具和验证器 (`diagnostic/tools/`, `diagnostic/utils/`, `diagnostic/validators.py`, `diagnostic/state_checker.py`)。
- 新增人机交互模块 (`diagnostic/hitl/`)。
- 新增提示词管理模块 (`diagnostic/prompts/`)。
- 新增物理拓扑数据库模块 (`topology/`)。
- 新增 FastAPI 后端 (`api/`)。
- 新增对应的单元测试 (`tests/`)。

## Impact
- Affected specs: 车辆电检报告智能诊断流程。
- Affected code: 主要新增 `diagnostic/`, `topology/`, `api/` 目录。不修改现有 `app/` 下的核心基础设施。

## ADDED Requirements
### Requirement: 诊断状态与编排
The system SHALL provide 完整的 `DiagnosticState` 状态定义及基于 LangGraph 的有向图编排，支持重试、降级和 Human-in-the-Loop 节点。

#### Scenario: Success case
- **WHEN** 用户提交电检报告
- **THEN** 系统通过图节点依次流转，经过解析、检索、推理、审计，最终输出 FinalReport。

### Requirement: 智能体协同
The system SHALL provide Parser解析、Surveyor拓扑查询、MCP知识检索、RCA根因分析、Auditor逻辑审计等智能体组件。

### Requirement: 动态 Prompt 与工具注入
The system SHALL provide PromptRegistry 和 ToolRegistry，支持运行时 Prompt 注入和工具的渐进式披露策略。

## MODIFIED Requirements
无

## REMOVED Requirements
无
