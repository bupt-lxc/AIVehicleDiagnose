# Tasks

* [x] Task 1: 基础架构与状态定义

  * [x] SubTask 1.1: 创建 `diagnostic/state.py`，定义 `DiagnosticState` 及相关数据模型

  * [x] SubTask 1.2: 创建 `diagnostic/config.py`，定义 `DiagnosticConfig`

  * [x] SubTask 1.3: 创建 `diagnostic/validators.py` 和 `diagnostic/state_checker.py`，实现输入和状态校验逻辑

* [x] Task 2: 核心注册表机制

  * [x] SubTask 2.1: 创建 `diagnostic/prompts/registry.py`，实现 Prompt 注册与动态注入

  * [x] SubTask 2.2: 创建 `diagnostic/tools/registry.py`，实现工具注册与渐进式披露控制器

  * [x] SubTask 2.3: 在 `diagnostic/prompts/` 目录下添加所有预定义的 Prompt 模板 (`parser_v1.py`, `rca_v1.py`, `auditor_v1.py`, `report_v1.py`)

* [x] Task 3: 拓扑数据库模块

  * [x] SubTask 3.1: 创建 `topology/` 目录，实现简单的 SQLite 数据库连接、模型和查询功能 (`database.py`, `models.py`, `queries.py`)

* [x] Task 4: 智能体组件实现

  * [x] SubTask 4.1: 实现 `diagnostic/agents/parser_agent.py` (包含本地和云端降级逻辑)

  * [x] SubTask 4.2: 实现 `diagnostic/agents/surveyor_agent.py` (集成拓扑查询)

  * [x] SubTask 4.3: 实现 `diagnostic/agents/mcp_librarian.py` (集成现有的 RAGClient)

  * [x] SubTask 4.4: 实现 `diagnostic/agents/rca_engine.py` (实现 ReAct 推理循环)

  * [x] SubTask 4.5: 实现 `diagnostic/agents/auditor_agent.py` (逻辑审计)

  * [x] SubTask 4.6: 实现 `diagnostic/agents/report_generator.py` (最终报告生成)

* [x] Task 5: Human-in-the-Loop (HITL) 机制

  * [x] SubTask 5.1: 实现 `diagnostic/hitl/handler.py` 和 `diagnostic/hitl/prompts.py`

* [x] Task 6: LangGraph 编排图实现

  * [x] SubTask 6.1: 创建 `diagnostic/graph.py`，串联所有节点并实现带有条件路由的 `AutoDiagnoseGraph`

* [x] Task 7: FastAPI 后端接口

  * [x] SubTask 7.1: 创建 `api/main.py` 及 `api/routes/` 目录下的 `diagnostic.py` 和 `health.py`

* [x] Task 8: 测试与验证

  * [x] SubTask 8.1: 编写核心模块和 Agent 的单元测试 (`tests/test_*.py`)

# Task Dependencies

* Task 2 depends on Task 1

* Task 3 depends on Task 1

* Task 4 depends on Task 1, Task 2, Task 3

* Task 5 depends on Task 1

* Task 6 depends on Task 4, Task 5

* Task 7 depends on Task 6

* Task 8 depends on Task 6

