# AI Diagnostic

基于 LangGraph 的车辆故障诊断引擎，使用 LLM (DeepSeek / 通义千问) 进行电检报告解析、拓扑构建、RAG 知识检索和 ReAct 根因推理。

## 架构

```
XML 电检报告
    │
    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Parser      │───▶│ DTC Focus    │───▶│  Surveyor    │
│  解析XML     │    │ 优先级筛选    │    │ 拓扑图构建    │
└──────────────┘    └──────────────┘    └──────────────┘
                                                │
    ┌──────────────┐    ┌──────────────┐        ▼
    │  Report      │◀───│  Auditor     │◀──┌──────────────┐
    │  Generator   │    │  逻辑审计     │   │  RCA Engine  │
    │  Markdown输出 │    └──────────────┘   │  ReAct 推理   │
    └──────────────┘                       └──────────────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │ MCP Librarian│
                                          │ 知识库检索    │
                                          └──────────────┘
```

## 功能

- **XML 电检报告解析**: 自动解析 ODX/PDX 格式的诊断报告，提取 ECU 状态、DTC 码、冻结帧
- **DTC 优先级筛选**: 根据严重程度、事件类型和用户上下文自动聚焦关键 DTC
- **车辆拓扑构建**: 从 SQLite 拓扑数据库查询 ECU 通信关系和跳数距离
- **ReAct 根因推理**: LLM 驱动的多轮推理循环，支持工具调用 (query_dtc, check_ecu_hop_distance 等)
- **HITL 人工介入**: 低置信度、审计冲突、报告审批节点支持交互式人工决策
- **审计检查**: 自动验证根因 ECU 拓扑一致性、DTC 覆盖率和证据闭环
- **双模式知识检索**: 支持 RAG 服务端检索和本地 DTC 数据库降级查询
- **多模型回退**: DeepSeek API 调用失败时自动尝试回退模型链

## 快速开始

### 环境要求

- Python >= 3.10
- SQLite 拓扑数据库 (`vehicle_info/vehicle_architecture.db`)

### 安装

```bash
pip install -r requirements.txt
```

### 配置

复制 `.env.example` 为 `.env` 并填写 API Key:

```env
# DeepSeek API (OpenAI 兼容)
ONLINE_LLM_BASE_URL=https://api.deepseek.com
ONLINE_LLM_API_KEY=sk-your-deepseek-api-key
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
ONLINE_LLM_MODEL=deepseek-v4-pro

# 或使用阿里云通义千问
# ONLINE_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# ONLINE_LLM_API_KEY=sk-your-dashscope-key
# ONLINE_LLM_MODEL=qwen-plus

# RAG 服务 (可选，不配置则使用本地 DTC 数据库降级)
RAG_SERVER_URL=http://localhost:7300

# 拓扑数据库路径
TOPOLOGY_DB_PATH=vehicle_info/vehicle_architecture.db
```

### 运行

```bash
# 交互模式 (默认)
python run_interactive.py testing/test1.xml

# 自动模式 (无需人工确认)
python run_interactive.py testing/test1.xml --auto

# 附加上下文描述
python run_interactive.py testing/test1.xml -c "车辆行驶中碰撞预警灯亮起"
```

### 输出

- `logs/diagnostic_report.md` — 结构化 Markdown 诊断报告
- `logs/interactive_diagnostic_report.json` — 完整诊断状态快照

## 项目结构

```
├── app/
│   ├── config.py              # 配置与 Settings 加载
│   ├── providers.py           # LLM Provider (OpenAI 兼容 + 多模型回退)
│   ├── diagnostic/
│   │   ├── agents/            # 各阶段 Agent (Parser, Surveyor, RCA, Auditor, Report)
│   │   ├── tools/             # 工具注册与实现 (DTC查询, 拓扑跳距等)
│   │   ├── prompts/           # Prompt 模板注册
│   │   ├── hitl/              # Human-in-the-Loop 处理器
│   │   ├── graph.py           # LangGraph 状态图定义
│   │   ├── state.py           # 诊断状态 TypedDict
│   │   ├── prioritizer.py     # DTC 优先级引擎
│   │   └── validators.py      # 输入验证
│   ├── topology/              # 拓扑数据库 ORM 与查询
│   └── rag_client.py          # RAG 服务客户端
├── tests/                     # 单元测试
├── vehicle_info/              # 拓扑 SQLite 数据库
├── testing/                   # 测试用 XML 电检报告
├── run_interactive.py         # 交互式主入口
└── requirements.txt
```

## 关键特性

### 跳过 RAG 离线运行

当 RAG 服务不可用时，`LocalMCPLibrarianAgent` 自动使用本地 DTC 数据库降级:

```python
# run_interactive.py 中已配置
class LocalMCPLibrarianAgent(MCPLibrarianAgent):
    async def retrieve(self, state):
        # 直接从本地 dtc_database 查询 DTC 定义
        ...
```

### DeepSeek 多模型回退

`app/providers.py` 中的 `OpenAICompatibleProvider` 在 API 返回模型不可用/配额耗尽等错误时，自动按以下优先级尝试回退模型:

1. 配置的 `ONLINE_LLM_MODEL`
2. `ONLINE_LLM_FALLBACK_MODELS` 环境变量指定的模型列表
3. DeepSeek: `deepseek-v4-flash` / 通义千问: `qwen-turbo → qwen-max → qwen-plus`

### HITL 自动批准

`--auto` 模式下，HITL (Human-in-the-Loop) 节点自动放行:
- `hitl_parse_confirm`: 跳过解析确认
- `hitl_low_confidence`: 直接进入审计
- `hitl_audit_conflict`: 跳过冲突处理
- `hitl_report_approval`: 自动批准报告

## License

Proprietary — Audi / VW Group internal tool.
