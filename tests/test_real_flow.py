
import asyncio
import os
import sys
import json
import logging
from pathlib import Path
import sqlite3
from collections import deque
import os
import sys

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.topology.models import TopologyGraphModel, ECUModel, ConnectionModel

# 1. 优先 Patch Topology，防止 SurveyorAgent 导入旧函数
import app.topology.queries

def patched_get_sub_topology(center_ecu_id: str, depth: int = 2) -> TopologyGraphModel:
    print(f"[TopologyPatch] Searching for center_ecu_id: '{center_ecu_id}' with depth {depth}")
    db_path = os.environ["TOPOLOGY_DB_PATH"]
    nodes = {}
    edges = []
    visited = set()
    MAX_VISITED_NODES = 1000
    
    queue = deque([(center_ecu_id, 0)])
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        while queue and len(visited) < MAX_VISITED_NODES:
            current_ecu, current_depth = queue.popleft()
            
            if current_ecu in visited or current_depth > depth:
                continue
                
            visited.add(current_ecu)
            
            # Query Node (ecu_info)
            cursor.execute("SELECT * FROM ecu_info WHERE ecu_address = ?", (current_ecu,))
            row = cursor.fetchone()
            if row:
                nodes[current_ecu] = ECUModel(
                    ecu_id=row['ecu_address'],
                    domain=row['ecu_domain'] or "Unknown",
                    power_segment="Unknown",
                    is_gateway=False
                )
            else:
                # print(f"[TopologyPatch] Node '{current_ecu}' not found in ecu_info")
                # Try creating placeholder anyway if it exists in topology
                nodes[current_ecu] = ECUModel(
                    ecu_id=current_ecu,
                    domain="Unknown",
                    power_segment="Unknown",
                    is_gateway=False
                )
            
            if current_depth < depth:
                # Query Edges (topology)
                cursor.execute("SELECT * FROM topology WHERE main_ecu = ? OR connected_ecu = ?", (current_ecu, current_ecu))
                rows = cursor.fetchall()
                # if not rows and current_depth == 0:
                #      print(f"[TopologyPatch] No edges found for '{current_ecu}'")
                     
                for row in rows:
                    source = row['main_ecu']
                    target = row['connected_ecu']
                    protocol = row['connection_protocols'] or "Unknown"
                    
                    # Clean protocol
                    if "CAN" in protocol: protocol = "CAN"
                    elif "LIN" in protocol: protocol = "LIN"
                    elif "FlexRay" in protocol: protocol = "FlexRay"
                    elif "Ethernet" in protocol or "ETH" in protocol: protocol = "ETH"
                    else: protocol = "Unknown"

                    edge = ConnectionModel(
                        source_id=str(source),
                        target_id=str(target),
                        protocol=protocol,
                        gateway_id=None
                    )
                    edges.append(edge)
                    
                    if source not in visited:
                        queue.append((source, current_depth + 1))
                    if target not in visited:
                        queue.append((target, current_depth + 1))
        
        conn.close()
    except Exception as e:
        logger.error(f"Topology query failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Deduplicate edges
    unique_edges = []
    seen = set()
    for edge in edges:
        identifier = f"{edge.source_id}-{edge.target_id}-{edge.protocol}"
        if identifier not in seen:
            seen.add(identifier)
            unique_edges.append(edge)
            
    return TopologyGraphModel(
        nodes=list(nodes.values()),
        edges=unique_edges,
        depth=depth,
        center_ecu=center_ecu_id
    )

app.topology.queries.get_sub_topology = patched_get_sub_topology

class MockHITLHandler:
    """模拟HITL交互处理器，自动通过所有检查"""
    def __init__(self):
        self.interaction_log = []
    
    def handle_parse_confirm(self, state):
        print("\n[HITL] 自动确认解析结果")
        self.interaction_log.append("parse_confirm: auto_confirm")
        return {"hitl_required": False, "hitl_question": "Confirm?", "hitl_options": ["A"]}
    
    def handle_low_confidence(self, state):
        print("\n[HITL] 自动处理低置信度 -> 继续")
        self.interaction_log.append("low_confidence: auto_proceed")
        return {"hitl_required": False, "hitl_question": "Continue?", "hitl_options": ["A"]}
    
    def handle_audit_conflict(self, state):
        print("\n[HITL] 自动处理审计冲突 -> 接受")
        self.interaction_log.append("audit_conflict: auto_accept")
        return {"hitl_required": False, "hitl_question": "Accept?", "hitl_options": ["A"]}
    
    def handle_report_approval(self, state):
        print("\n[HITL] 自动批准报告")
        self.interaction_log.append("report_approval: auto_approve")
        return {"hitl_required": False, "hitl_question": "Approve?", "hitl_options": ["A"]}

# 2. 导入其他模块
from app.config import load_settings
from app.llm_manager import LLMManager
from app.rag_client import RAGClient
from app.diagnostic.graph import AutoDiagnoseGraph
from app.diagnostic.state import create_initial_state, RAGKnowledge
from app.diagnostic.agents import (
    ParserAgent, SurveyorAgent, MCPLibrarianAgent, 
    RCAEngine, AuditorAgent, ReportGenerator
)

class PatchedReportGenerator(ReportGenerator):
    """处理 Context Size 限制的 ReportGenerator"""
    def generate(self, state: dict) -> dict:
        # 截断输入数据，只保留高优先级的 DTC
        # 注意：这里我们需要修改 state 的副本，不能影响后续流程（虽然 ReportGenerator 是最后一步）
        # 但为了简单，我们直接修改 state，反正之后也没啥了
        
        structured_data = state.get("structured_data", {})
        dtc_codes = structured_data.get("dtc_codes", [])
        
        # 按优先级排序并只保留前 30 个
        sorted_dtcs = sorted(dtc_codes, key=lambda x: x.get("priority_score", 0), reverse=True)
        
        # 创建一个新的 state 用于生成报告，避免污染原始 state
        temp_state = state.copy()
        temp_struct = structured_data.copy()
        
        if len(sorted_dtcs) > 30:
            print(f"[ReportGenerator] Truncating DTCs from {len(dtc_codes)} to 30 for LLM context limit")
            temp_struct["dtc_codes"] = sorted_dtcs[:30]
            
        # 同时清空日志或 freeze frames
        if "raw_text_segments" in temp_struct:
            temp_struct["raw_text_segments"] = []
            
        temp_state["structured_data"] = temp_struct
        
        return super().generate(temp_state)

class PatchedRCAEngine(RCAEngine):
    """处理 Context Size 限制的 RCAEngine"""
    async def analyze(self, state: dict) -> dict:
        structured_data = state.get("structured_data", {})
        dtc_codes = structured_data.get("dtc_codes", [])
        
        # 按优先级排序并只保留前 30 个 (兼顾上下文与覆盖率)
        sorted_dtcs = sorted(dtc_codes, key=lambda x: x.get("priority_score", 0), reverse=True)
        
        temp_state = state.copy()
        temp_struct = structured_data.copy()
        
        if len(sorted_dtcs) > 30:
            print(f"[RCAEngine] Truncating DTCs from {len(dtc_codes)} to 30 for LLM context limit")
            temp_struct["dtc_codes"] = sorted_dtcs[:30]
            
        # 清空原始文本，太占空间
        if "raw_text_segments" in temp_struct:
            temp_struct["raw_text_segments"] = []
            
        temp_state["structured_data"] = temp_struct
        
        result = await super().analyze(temp_state)
        print(f"[PatchedRCAEngine] Return analysis_steps length: {len(result.get('analysis_steps', []))}")
        print(f"[PatchedRCAEngine] Return root_cause: {result.get('preliminary_root_cause')}")
        return result

from app.diagnostic.tools import dtc_database
from app.diagnostic.prompts.registry import PromptRegistry
from app.diagnostic.tools.registry import ToolRegistry
from app.diagnostic.prompts.parser_v1 import parser_prompt
from app.diagnostic.prompts.rca_v1 import rca_prompt
from app.diagnostic.prompts.auditor_v1 import auditor_prompt
from app.diagnostic.prompts.report_v1 import report_prompt

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 屏蔽 langgraph 的序列化警告
logging.getLogger("langgraph.checkpoint.serde.jsonplus").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# 设置拓扑数据库路径
_PROJECT_ROOT = Path(__file__).parent.parent
os.environ["TOPOLOGY_DB_PATH"] = str(_PROJECT_ROOT / "vehicle_info" / "vehicle_architecture.db")


from app.diagnostic.tools.xml_report_parser import parse_diagnostic_report as xml_parser, convert_to_structured_data

class EnrichedParserAgent(ParserAgent):
    """增强版 ParserAgent，解析后自动查库补全 ECU 和 描述信息"""
    
    def parse(self, state):
        # 1. 尝试直接解析 XML 文件
        raw_input = state.get("raw_input", "")
        result = None
        
        if raw_input.strip().endswith('.xml') and os.path.exists(raw_input.strip()):
            logger.info(f"[EnrichedParser] 检测到 XML 文件输入: {raw_input}")
            parsed_report = xml_parser(raw_input.strip())
            if parsed_report:
                structured_data = convert_to_structured_data(parsed_report)
                
                # 优先级评定与过滤
                user_context = state.get("user_context", "")
                threshold_str = self._config.get("dtc_filter_threshold", "Low")
                try:
                    from app.diagnostic.prioritizer import DTCSeverity
                    threshold = DTCSeverity(threshold_str)
                except ValueError:
                    from app.diagnostic.prioritizer import DTCSeverity
                    threshold = DTCSeverity.LOW
                    
                processed_dtcs = self._prioritizer.process_dtcs(structured_data["dtc_codes"], user_context, threshold)
                structured_data["dtc_codes"] = processed_dtcs
                
                result = {
                    "structured_data": structured_data,
                    "parse_success": True,
                    "parse_error": None,
                    "retry_state": {**state.get("retry_state", {}), "parser_retries": 0},
                    "current_node": self.NODE_NAME,
                }
        
        # 2. 如果不是 XML 或者解析失败，调用原始解析
        if not result:
            result = super().parse(state)
        
        if not result.get("parse_success"):
            return result
            
        structured_data = result.get("structured_data", {})
        dtc_codes = structured_data.get("dtc_codes", [])
        
        logger.info(f"[EnrichedParser] 解析出 {len(dtc_codes)} 个 DTC，开始补全信息...")
        
        # 2. 查库补全
        updated_dtcs = []
        for dtc in dtc_codes:
            code = dtc.get("code")
            if not code:
                continue
                
            query_keys = []
            fehler_typ = ((dtc.get("raw_data") or {}).get("fehler_typ") or "").strip().upper()
            if fehler_typ:
                query_keys.append(fehler_typ)
            if code:
                query_keys.append(code)

            if not dtc.get("description") or not dtc.get("ecu_id"):
                try:
                    query_result = None
                    for key in query_keys:
                        query_result = dtc_database.query_dtc(key)
                        if query_result.get("success") and query_result.get("results"):
                            break

                    if query_result and query_result.get("success") and query_result.get("results"):
                        info = query_result["results"][0]
                        if not dtc.get("description"):
                            dtc["description"] = info.get("description")
                        if not dtc.get("ecu_id"):
                            dtc["ecu_id"] = info.get("ecu_id")
                        dtc["ecu_name"] = info.get("ecu_longname")
                        dtc["dtc_level"] = info.get("dtc_level")
                        logger.info(f"[EnrichedParser] 补全 DTC {code}: ECU={dtc.get('ecu_id')}")
                except Exception as e:
                    logger.warning(f"[EnrichedParser] 查询 DTC {code} 失败: {e}")
            
            updated_dtcs.append(dtc)
            
        structured_data["dtc_codes"] = updated_dtcs
        
        # 3. 顺便更新 ECU States (如果有新的 ECU ID 出现)
        existing_ecu_ids = set(e.get("ecu_id") for e in structured_data.get("ecu_states", []))
        new_ecu_states = structured_data.get("ecu_states", [])
        
        for dtc in updated_dtcs:
            ecu_id = dtc.get("ecu_id")
            if ecu_id and ecu_id not in existing_ecu_ids:
                new_ecu_states.append({
                    "ecu_id": ecu_id,
                    "status": "Unknown", # 默认为未知，等待后续分析
                    "domain": "Unknown"
                })
                existing_ecu_ids.add(ecu_id)
        
        structured_data["ecu_states"] = new_ecu_states
        result["structured_data"] = structured_data
        
        return result

class LocalMCPLibrarianAgent(MCPLibrarianAgent):
    """本地版 MCPLibrarianAgent，不查 RAG，查本地库"""
    
    async def retrieve(self, state):
        logger.info("[LocalMCPLibrarian] 开始检索知识 (Local Mode)...")
        
        structured_data = state.get("structured_data", {})
        dtc_codes = [d["code"] for d in structured_data.get("dtc_codes", []) if d.get("code")]
        
        dtc_definitions = []
        
        # 查本地库获取定义
        for dtc_item in structured_data.get("dtc_codes", []):
            code = dtc_item.get("code")
            if not code:
                continue
            try:
                query_keys = []
                fehler_typ = ((dtc_item.get("raw_data") or {}).get("fehler_typ") or "").strip().upper()
                if fehler_typ:
                    query_keys.append(fehler_typ)
                query_keys.append(code)

                res = None
                for key in query_keys:
                    res = dtc_database.query_dtc(key, limit=3)
                    if res.get("success") and res.get("results"):
                        break

                if res.get("success") and res.get("results"):
                    merged_lines = []
                    for info in res["results"][:3]:
                        line = f"Code: {info['dtc_code']}\nDescription: {info['description']}\nECU: {info['ecu_longname']} ({info['ecu_id']})\nLevel: {info['dtc_level']}"
                        if info.get("environment_data"):
                            line += f"\nEnvironment Data: {json.dumps(info['environment_data'], ensure_ascii=False)}"
                        merged_lines.append(line)

                    dtc_definitions.append({
                        "code": code,
                        "definition": "\n---\n".join(merged_lines),
                        "source": "local_database"
                    })
            except Exception as e:
                logger.warning(f"[LocalMCPLibrarian] 查询 {code} 失败: {e}")
        
        # 构造结果
        rag_knowledge = RAGKnowledge(
            dtc_definitions=dtc_definitions,
            case_studies=[], # 暂无案例
            repair_procedures=[] # 暂无维修流程
        )
        
        return {
            "rag_knowledge": rag_knowledge,
            "current_node": self.NODE_NAME,
        }

async def run_real_diagnostic():
    print("=" * 70)
    print("真实环境全流程诊断测试")
    print("=" * 70)
    
    settings = load_settings()
    
    # 强制使用本地 LLM 配置
    # 注意：Providers 会读取环境变量或 settings
    # 假设 .env 已经配置好 LOCAL_LLM_BASE_URL
    
    # 初始化组件
    # 1. Prompt Registry
    prompt_registry = PromptRegistry()
    prompt_registry.register(parser_prompt)
    prompt_registry.register(rca_prompt)
    prompt_registry.register(auditor_prompt)
    prompt_registry.register(report_prompt)
    
    # 2. Tool Registry
    tool_registry = ToolRegistry()
    # 注册 dtc 工具供 RCA 引擎使用
    dtc_database.register_tools(tool_registry)
    from app.diagnostic.tools import topology_hop_checker, vehicle_architecture_tool
    topology_hop_checker.register_tools(tool_registry)
    vehicle_architecture_tool.register_tools(tool_registry)
    
    # 3. Agents
    # 使用 Cloud LLM Provider
    settings.default_profile = "online"
    os.environ["ONLINE_LLM_FALLBACK_MODELS"] = ",".join(
        [
            "qwen3-max-2025-09-23",
            "qwen-plus-2025-12-01",
            "qwen-turbo-latest",
            "qwen2.5-32b-instruct",
            "qwen-plus-2025-04-28",
            "qwen-plus-2025-01-25",
        ]
    )
    
    from app.providers import OpenAICompatibleProvider
    cloud_llm = OpenAICompatibleProvider(profile=settings.online_profile, timeout_seconds=120)
    
    parser_agent = EnrichedParserAgent(
        cloud_llm=cloud_llm, 
        prompt_registry=prompt_registry
    )
    
    surveyor_agent = SurveyorAgent(config={"topology_depth": 1})
    
    mcp_librarian = LocalMCPLibrarianAgent(rag_client=None) # 不需要 RAG Client
    
    rca_engine = PatchedRCAEngine(
        llm=cloud_llm,
        prompt_registry=prompt_registry,
        tool_registry=tool_registry
    )
    
    auditor_agent = AuditorAgent(
        llm=cloud_llm,
        prompt_registry=prompt_registry
    )
    
    report_generator = PatchedReportGenerator(
        llm=cloud_llm,
        prompt_registry=prompt_registry
    )
    
    hitl_handler = MockHITLHandler()
    
    # 4. Graph
    db_path = settings.graph_checkpoint_db
    
    async with AutoDiagnoseGraph(
        parser_agent=parser_agent,
        surveyor_agent=surveyor_agent,
        mcp_librarian=mcp_librarian,
        rca_engine=rca_engine,
        auditor_agent=auditor_agent,
        report_generator=report_generator,
        hitl_handler=hitl_handler,
        db_path=db_path
    ) as graph:
        
        xml_path = str(_PROJECT_ROOT / "testing" / "test1.xml")
        user_context = "该车自动停车有问题，用户反映停车辅助功能异常"
        
        initial_state = create_initial_state(
            raw_input=xml_path,
            user_context=user_context
        )
        
        print(f"Thread ID: {initial_state['thread_id']}")
        
        try:
            final_state = await graph.run(initial_state)
            
            print("\n" + "=" * 70)
            print("诊断完成")
            print("=" * 70)
            
            # 输出结果摘要
            print(f"解析成功: {final_state.get('parse_success')}")
            
            struct = final_state.get("structured_data", {})
            print(f"DTC 数量: {len(struct.get('dtc_codes', []))}")
            
            topo = final_state.get("topology_graph")
            if topo:
                print(f"拓扑图节点数: {len(topo.get('nodes', []))}")
                print(f"拓扑图边数: {len(topo.get('edges', []))}")
            else:
                print("拓扑图为空")
                
            report = final_state.get("final_report")
            if report:
                print("\n最终报告:")
                print(f"根因: {report.get('root_cause')}")
                print(f"置信度: {report.get('confidence_score')}")
                print(f"受影响 ECU: {report.get('affected_ecus')}")
            
            # 保存结果
            report_path = Path("logs/real_diagnostic_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(final_state, f, ensure_ascii=False, indent=2, default=str)
            print(f"报告已保存: {report_path.absolute()}")
            
        except Exception as e:
            logger.exception("运行失败")
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_real_diagnostic())
