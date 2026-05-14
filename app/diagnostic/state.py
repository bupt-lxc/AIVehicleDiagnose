# app/diagnostic/state.py

import uuid
from typing import TypedDict, Annotated, Literal, Any
from datetime import datetime, timezone
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class DTCCode(TypedDict):
    code: str                          # DTC 代码，如 "P0135"
    status: Literal["Active", "Stored", "Pending"]
    description: str | None            # DTC 描述
    ecu_id: str | None                 # 关联的 ECU ID
    severity: str | None               # 严重等级: Critical, High, Medium, Low, Unknown
    priority_score: int | None         # 优先级分数


class ECUState(TypedDict):
    ecu_id: str                        # ECU 标识符
    domain: str                        # 域控: Powertrain/Chassis/Body/Infotainment/ADAS
    status: Literal["Online", "Offline", "Degraded"]
    communication_status: str | None   # 通信状态
    power_segment: str | None          # 电源段


class FreezeFrame(TypedDict):
    dtc_code: str                      # 关联的 DTC
    timestamp: str | None              # 时间戳
    rpm: int | None                    # 发动机转速
    speed: int | None                  # 车速
    temperature: float | None          # 温度
    voltage: float | None              # 电压
    additional_data: dict[str, Any]    # 其他数据


class StructuredData(TypedDict):
    dtc_codes: list[DTCCode]           # DTC 列表
    ecu_states: list[ECUState]         # ECU 状态列表
    freeze_frames: list[FreezeFrame]   # 冻结帧列表
    raw_text_segments: list[str]       # 原始文本片段


class TopologyNode(TypedDict):
    ecu_id: str                        # ECU ID
    domain: str                        # 域
    power_segment: str                 # 电源段
    gateway: bool                      # 是否为网关


class TopologyEdge(TypedDict):
    source_id: str                     # 源 ECU ID
    target_id: str                     # 目标 ECU ID
    protocol: Literal["CAN", "LIN", "ETH", "FlexRay"]
    gateway_id: str | None             # 经过的网关


class TopologyGraph(TypedDict):
    nodes: list[TopologyNode]          # 节点列表
    edges: list[TopologyEdge]          # 边列表
    depth: int                         # 拓扑深度
    center_ecu: str                    # 中心 ECU ID


class RAGKnowledge(TypedDict):
    dtc_definitions: list[dict]        # DTC 定义
    case_studies: list[dict]           # 历史案例
    repair_procedures: list[str]       # 维修流程


class EvidenceGraphNode(TypedDict):
    id: str
    node_type: Literal["DTC", "ECU", "PROTOCOL", "SYMPTOM", "TIMELINE", "HISTORICAL_CASE", "ROOT_CAUSE_CANDIDATE"]
    label: str
    attributes: dict[str, Any]


class EvidenceGraphEdge(TypedDict):
    source_id: str
    target_id: str
    relation: Literal["supports", "contradicts", "depends_on", "excluded_by"]
    evidence: str
    attributes: dict[str, Any]


class EvidenceGraph(TypedDict):
    nodes: list[EvidenceGraphNode]
    edges: list[EvidenceGraphEdge]


class HITLFeedback(TypedDict):
    point: str                         # HITL 点
    question: str                      # 问题内容
    options: list[str]                 # 选项列表
    user_choice: str                   # 用户选择
    user_input: str | None             # 用户额外输入
    timestamp: str                     # 时间戳
    timed_out: bool                    # 是否超时


class RetryState(TypedDict):
    parser_retries: int                # Parser 重试次数
    rca_retries: int                   # RCA 重试次数
    audit_retries: int                 # Audit 重试次数
    revision_count: int                # 报告修改次数
    invalid_input_count: int           # 无效输入次数


class FinalReport(TypedDict):
    root_cause: str                    # 根因描述
    confidence_score: float            # 置信度 (0.0-1.0)
    affected_ecus: list[str]           # 受影响的 ECU 列表
    evidence: list[str]                # 证据链
    recommendations: list[str]         # 维修建议
    risks: list[str]                   # 风险提示
    audit_trail: list[str]             # 审计轨迹
    human_decisions: list[HITLFeedback]  # 人工决策记录
    pending_confirmation: bool         # 是否待确认
    confidence_breakdown: dict[str, float]  # 置信度分解
    hypothesis_candidates: list[dict[str, Any]]  # 并行根因候选
    rejected_hypotheses: list[dict[str, Any]]    # 显式排除树
    evidence_graph: EvidenceGraph              # 证据图谱


class DiagnosticState(TypedDict):
    # === 输入数据 ===
    raw_input: str                     # 原始电检文本/文件路径
    user_context: str                  # 用户描述症状
    
    # === 解析结果 ===
    structured_data: StructuredData    # 解析后的结构化数据
    parse_success: bool                # 解析是否成功
    parse_error: str | None            # 解析错误信息
    all_dtcs: list[DTCCode]            # 报告级 DTC 全量池
    focused_dtcs: list[DTCCode]        # 当前重点 DTC 池
    deferred_dtcs: list[DTCCode]       # 暂缓 DTC 池
    dtc_selection_log: list[dict[str, Any]]  # DTC筛选审计记录
    
    # === 拓扑数据 ===
    topology_graph: TopologyGraph | None  # 相关的物理连接子图
    
    # === 知识检索 ===
    rag_knowledge: RAGKnowledge | None    # MCP 返回的维修策略
    
    # === 推理结果 ===
    analysis_steps: list[str]          # 推理轨迹 (CoT)
    confidence_score: float            # 置信度
    confidence_breakdown: dict[str, float]  # 置信度分解
    confidence_history: list[dict[str, Any]]  # 置信度演进记录
    preliminary_root_cause: str | None # 初步根因
    hypothesis_queue: list[dict[str, Any]]    # 假设队列
    hypothesis_candidates: list[dict[str, Any]]  # 并行候选假设
    rejected_hypotheses: list[dict[str, Any]]    # 排除假设树
    evidence_graph: EvidenceGraph             # 证据图
    rca_iteration: int                         # RCA迭代轮次
    
    # === 审计结果 ===
    audit_log: list[str]               # 审计反馈记录
    audit_passed: bool                 # 审计是否通过
    conflicts: list[str]               # 矛盾点列表
    
    # === 最终报告 ===
    final_report: FinalReport | None   # 最终输出的报告
    report_approved: bool              # 报告是否批准
    follow_up_question: str            # 报告后续交互问题
    conversation_suggestion: list[str] # 后续建议方向
    
    # === Human-in-the-Loop ===
    hitl_required: bool                # 是否需要人工介入
    current_hitl_point: str | None     # 当前 HITL 点
    hitl_question: str                 # HITL 问题
    hitl_options: list[str]            # HITL 选项
    hitl_feedbacks: list[HITLFeedback] # HITL 反馈历史
    
    # === 人工输入 ===
    priority_dtcs: list[str]           # 优先分析的 DTC
    human_hint: str | None             # 人工提示
    user_description: str | None       # 用户额外描述
    
    # === 重试状态 ===
    retry_state: RetryState            # 重试状态
    
    # === 风险标记 ===
    risk_flags: list[str]              # 风险标记
    confidence_warning: bool           # 置信度警告
    conflict_ignored: bool             # 矛盾是否被忽略
    
    # === 执行状态 ===
    current_node: str                  # 当前节点名称
    error_message: str | None          # 错误信息
    exit_code: int                     # 退出码: 0=成功, 1=失败
    transition_log: list[dict[str, Any]]      # 状态迁移轨迹
    operator_trace: list[dict[str, Any]]      # 用户可见操作轨迹
    internal_reasoning: list[dict[str, Any]]  # 内部推理记录
    degraded_mode: bool                        # 本地降级模式
    knowledge_degraded: bool                   # 知识检索降级模式
    
    # === 元数据 ===
    thread_id: str                     # 会话 ID
    run_id: str                        # 运行 ID
    started_at: str                    # 开始时间
    finished_at: str | None            # 结束时间
    
    # === LangChain 消息 ===
    messages: Annotated[list[AnyMessage], add_messages]


def create_initial_state(
    raw_input: str,
    user_context: str = "",
    thread_id: str | None = None,
) -> DiagnosticState:
    """创建初始状态"""
    
    return DiagnosticState(
        raw_input=raw_input,
        user_context=user_context,
        structured_data=StructuredData(
            dtc_codes=[],
            ecu_states=[],
            freeze_frames=[],
            raw_text_segments=[],
        ),
        parse_success=False,
        parse_error=None,
        all_dtcs=[],
        focused_dtcs=[],
        deferred_dtcs=[],
        dtc_selection_log=[],
        topology_graph=None,
        rag_knowledge=None,
        analysis_steps=[],
        confidence_score=0.0,
        confidence_breakdown={
            "evidence_sufficiency": 0.0,
            "topology_consistency": 0.0,
            "temporal_consistency": 0.0,
            "historical_prior_match": 0.0,
            "aggregated_confidence": 0.0,
        },
        confidence_history=[],
        preliminary_root_cause=None,
        hypothesis_queue=[],
        hypothesis_candidates=[],
        rejected_hypotheses=[],
        evidence_graph=EvidenceGraph(nodes=[], edges=[]),
        rca_iteration=0,
        audit_log=[],
        audit_passed=False,
        conflicts=[],
        final_report=None,
        report_approved=False,
        follow_up_question="",
        conversation_suggestion=[],
        hitl_required=False,
        current_hitl_point=None,
        hitl_question="",
        hitl_options=[],
        hitl_feedbacks=[],
        priority_dtcs=[],
        human_hint=None,
        user_description=None,
        retry_state=RetryState(
            parser_retries=0,
            rca_retries=0,
            audit_retries=0,
            revision_count=0,
            invalid_input_count=0,
        ),
        risk_flags=[],
        confidence_warning=False,
        conflict_ignored=False,
        current_node="input_validator",
        error_message=None,
        exit_code=0,
        transition_log=[],
        operator_trace=[],
        internal_reasoning=[],
        degraded_mode=False,
        knowledge_degraded=False,
        thread_id=thread_id or f"diag-{uuid.uuid4().hex[:12]}",
        run_id=uuid.uuid4().hex,
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
        messages=[],
    )
