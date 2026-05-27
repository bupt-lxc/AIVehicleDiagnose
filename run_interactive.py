"""
交互式车辆故障诊断工具 v2
=========================
完整体验 AutoDiagnoseGraph 的全流程分析：
- Parser → DTC Focus → Surveyor → MCP Librarian → RCA Engine → Auditor → Report Generator
- 支持 HITL 人工交互（选项 + 补充内容输入）
- 实时流式显示RCA推理过程、工具调用、Observation
"""

import asyncio
import os
import sys
import json
import time
from pathlib import Path

# Fix Windows console encoding for Chinese characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger

from app.config import load_settings
from app.diagnostic.state import create_initial_state
from app.diagnostic.agents import ParserAgent, SurveyorAgent, MCPLibrarianAgent, RCAEngine, AuditorAgent, ReportGenerator
from app.diagnostic.agents.rca_engine import RCAConfig
from app.diagnostic.tools.registry import ToolRegistry
from app.diagnostic.tools import dtc_database, topology_hop_checker, vehicle_architecture_tool
from app.diagnostic.tools.xml_report_parser import parse_diagnostic_report, convert_to_structured_data
from app.diagnostic.prompts.registry import PromptRegistry
from app.diagnostic.prompts.parser_v1 import parser_prompt
from app.diagnostic.prompts.rca_v1 import rca_prompt
from app.diagnostic.prompts.auditor_v1 import auditor_prompt
from app.diagnostic.prompts.report_v1 import report_prompt
from app.diagnostic.graph import AutoDiagnoseGraph
from app.providers import OpenAICompatibleProvider
from app.diagnostic.prioritizer import DTCSeverity

_PROJECT_ROOT = Path(__file__).parent

# ANSI
G = "\033[92m"
Y = "\033[93m"
B = "\033[94m"
C = "\033[96m"
M = "\033[95m"
R = "\033[91m"
D = "\033[1m"
X = "\033[0m"


def hdr(t):
    print(f"\n{D}{'=' * 70}{X}\n{D}{t}{X}\n{D}{'=' * 70}{X}")


def nd(name, desc):
    print(f"\n{B}{'─' * 60}{X}")
    print(f"{D}[节点]{X} {C}{name}{X}")
    print(f"       {desc}")
    print(f"{B}{'─' * 60}{X}")


def ok(name, status, detail=None):
    icon = G + "[✓]" + X if status == "ok" else R + "[✗]" + X
    print(f"  {icon} {name}: {status}")
    if detail:
        print(f"  {detail}")


def rca_stream(trace_item, iteration):
    """实时流式打印RCA推理过程"""
    t = trace_item.get("type", "")

    if t == "preload_observation":
        detail = trace_item.get("detail", "")
        if isinstance(detail, str) and "Thought:" in detail:
            thought = detail.replace("Thought: ", "").strip()
            print(f"  {M}[预加载]{X} Iter {iteration}: {thought[:120]}...")
    elif t == "react_thought":
        thought = trace_item.get("content", "")[:150]
        print(f"  {M}[思考]{X} Iter {iteration}: {thought}")
    elif t == "react_action":
        tool = trace_item.get("tool", "")
        inp = trace_item.get("input", {})
        # 简洁显示
        if isinstance(inp, dict):
            inp_str = ", ".join(f"{k}={str(v)[:40]}" for k, v in list(inp.items())[:3])
        else:
            inp_str = str(inp)[:80]
        print(f"  {Y}[工具→]{X} Iter {iteration}: {tool}({inp_str})")
    elif t == "tool_call":
        tool = trace_item.get("tool", "")
        obs = trace_item.get("observation", "")
        # 截断Observration
        obs_short = obs[:150]
        if "success" in obs.lower() and "False" in obs:
            print(f"    {R}[查询失败]{X} {obs_short}")
        else:
            # 提取关键信息
            if "dtc_level" in obs:
                level = obs.split("dtc_level")[1].split("\n")[0].strip().split(":")[-1].strip()
                print(f"    {G}[结果]{X} DTC Level: {level}")
            elif "topology" in tool.lower() or "hop" in tool.lower():
                print(f"    {G}[结果]{X} {obs_short[:120]}")
            elif len(obs) > 10:
                print(f"    {G}[Observation]{X} {obs_short[:120]}...")
    elif t == "recall_deferred_dtcs":
        print(f"  {C}[召回]{X} 召回延迟DTC: ECU={trace_item.get('input', {}).get('ecu_id', '?')}")
    elif t == "react_final_answer":
        rc = trace_item.get("root_cause", "")[:200]
        conf = trace_item.get("confidence", 0)
        print(f"\n  {G}[最终结论]{X} 置信度: {conf:.0%}")
        print(f"  根因: {rc}")
    elif t == "hypothesis_candidates":
        items = trace_item.get("items", [])
        if items:
            print(f"\n  {C}[假设候选]{X} {len(items)} 个")
            for item in items[:3]:
                print(f"    - {item.get('ecu_id', '?')}: {item.get('description', '')[:80]}")
    elif t == "rejected_hypotheses":
        items = trace_item.get("items", [])
        if items:
            print(f"  {R}[排除假设]{X} {len(items)} 个")
            for item in items[:2]:
                print(f"    - {item.get('ecu_id', '?')}: {item.get('reason', '')[:80]}")


class EnrichedParserAgent(ParserAgent):
    def parse(self, state):
        raw_input = state.get("raw_input", "")
        result = None
        if raw_input.strip().endswith('.xml') and os.path.exists(raw_input.strip()):
            logger.info(f"[EnrichedParser] XML: {raw_input}")
            parsed_report = parse_diagnostic_report(raw_input.strip())
            if parsed_report:
                structured_data = convert_to_structured_data(parsed_report)
                user_context = state.get("user_context", "")
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
        if not result:
            result = super().parse(state)
        if not result.get("parse_success"):
            return result

        structured_data = result.get("structured_data", {})
        dtc_codes = structured_data.get("dtc_codes", [])
        logger.info(f"[EnrichedParser] {len(dtc_codes)} DTCs, enriching...")

        updated = []
        for dtc in dtc_codes:
            code = dtc.get("code")
            if not code:
                updated.append(dtc)
                continue
            query_keys = []
            ft = ((dtc.get("raw_data") or {}).get("fehler_typ") or "").strip().upper()
            if ft:
                query_keys.append(ft)
            query_keys.append(code)
            if not dtc.get("description") or not dtc.get("ecu_id"):
                try:
                    qr = None
                    for key in query_keys:
                        qr = dtc_database.query_dtc(key)
                        if qr.get("success") and qr.get("results"):
                            break
                    if qr and qr.get("success") and qr.get("results"):
                        info = qr["results"][0]
                        if not dtc.get("description"):
                            dtc["description"] = info.get("description")
                        if not dtc.get("ecu_id"):
                            dtc["ecu_id"] = info.get("ecu_id")
                        dtc["ecu_name"] = info.get("ecu_longname")
                        dtc["dtc_level"] = info.get("dtc_level")
                except Exception as e:
                    logger.warning(f"[EnrichedParser] {code}: {e}")
            updated.append(dtc)

        structured_data["dtc_codes"] = updated
        existing_ecu_ids = set(e.get("ecu_id") for e in structured_data.get("ecu_states", []))
        for dtc in updated:
            ecu_id = dtc.get("ecu_id")
            if ecu_id and ecu_id not in existing_ecu_ids:
                structured_data.setdefault("ecu_states", []).append({"ecu_id": ecu_id, "status": "Unknown", "domain": "Unknown"})
                existing_ecu_ids.add(ecu_id)

        result["structured_data"] = structured_data
        return result


class LocalMCPLibrarianAgent(MCPLibrarianAgent):
    async def retrieve(self, state):
        logger.info("[LocalMCPLibrarian] 检索知识 (Local)...")
        structured_data = state.get("structured_data", {})
        dtc_definitions = []
        for dtc_item in structured_data.get("dtc_codes", []):
            code = dtc_item.get("code")
            if not code:
                continue
            try:
                query_keys = []
                ft = ((dtc_item.get("raw_data") or {}).get("fehler_typ") or "").strip().upper()
                if ft:
                    query_keys.append(ft)
                query_keys.append(code)
                res = None
                for key in query_keys:
                    res = dtc_database.query_dtc(key, limit=3)
                    if res.get("success") and res.get("results"):
                        break
                if res.get("success") and res.get("results"):
                    merged = []
                    for info in res["results"][:3]:
                        line = f"Code: {info['dtc_code']}\nDesc: {info['description']}\nECU: {info['ecu_longname']} ({info['ecu_id']})\nLevel: {info['dtc_level']}"
                        if info.get("environment_data"):
                            line += f"\nEnv: {json.dumps(info['environment_data'], ensure_ascii=False)}"
                        merged.append(line)
                    dtc_definitions.append({"code": code, "definition": "\n---\n".join(merged), "source": "local"})
            except Exception as e:
                logger.warning(f"[LocalMCPLibrarian] {code}: {e}")

        from app.diagnostic.state import RAGKnowledge
        rag_knowledge = RAGKnowledge(dtc_definitions=dtc_definitions, case_studies=[], repair_procedures=[])
        return {"rag_knowledge": rag_knowledge, "current_node": self.NODE_NAME}


async def run_interactive(xml_path: str, user_context: str = "", auto: bool = False):
    hdr("初始化诊断引擎")

    # Settings
    settings = load_settings()
    settings.default_profile = "online"
    if "deepseek" not in settings.online_profile.base_url.lower():
        os.environ["ONLINE_LLM_FALLBACK_MODELS"] = ",".join([
            "qwen3-max-2025-09-23", "qwen-plus-2025-12-01", "qwen-turbo-latest",
            "qwen2.5-32b-instruct", "qwen-plus-2025-04-28", "qwen-plus-2025-01-25",
        ])

    # Topology DB path
    topology_db = _PROJECT_ROOT / "vehicle_info" / "vehicle_architecture.db"
    os.environ["TOPOLOGY_DB_PATH"] = str(topology_db)
    from app.topology.database import ConnectionPool
    ConnectionPool._instance = None

    # Registries
    prompt_registry = PromptRegistry()
    prompt_registry.register(parser_prompt)
    prompt_registry.register(rca_prompt)
    prompt_registry.register(auditor_prompt)
    prompt_registry.register(report_prompt)

    tool_registry = ToolRegistry()
    dtc_database.register_tools(tool_registry)
    topology_hop_checker.register_tools(tool_registry)
    vehicle_architecture_tool.register_tools(tool_registry)
    print(f"  {G}[✓]{X} Prompt & 工具注册完成")

    cloud_llm = OpenAICompatibleProvider(profile=settings.online_profile, timeout_seconds=120)
    print(f"  {G}[✓]{X} LLM (online): {settings.online_profile.model}")

    # Agents
    parser_agent = EnrichedParserAgent(cloud_llm=cloud_llm, prompt_registry=prompt_registry)
    surveyor_agent = SurveyorAgent(config={"topology_depth": 1})
    mcp_librarian = LocalMCPLibrarianAgent(rag_client=None)
    rca_engine = RCAEngine(
        llm=cloud_llm,
        prompt_registry=prompt_registry,
        tool_registry=tool_registry,
        config=RCAConfig(verbose=True, timeout_seconds=300.0),
    )
    auditor_agent = AuditorAgent(llm=cloud_llm, prompt_registry=prompt_registry)
    report_generator = ReportGenerator(llm=cloud_llm, prompt_registry=prompt_registry)
    print(f"  {G}[✓]{X} Agent 初始化完成")

    graph = AutoDiagnoseGraph(
        parser_agent=parser_agent, surveyor_agent=surveyor_agent,
        mcp_librarian=mcp_librarian, rca_engine=rca_engine,
        auditor_agent=auditor_agent, report_generator=report_generator,
        hitl_handler=None, db_path=settings.graph_checkpoint_db
    )
    print(f"  {G}[✓]{X} 诊断图构建完成")

    # === Run: 使用编译 Graph 的真正路由 ===
    async with graph as g:
        state = create_initial_state(raw_input=str(xml_path), user_context=user_context)

        hdr("开始诊断分析")
        print(f"  {D}文件:{X} {xml_path}")
        print(f"  {D}描述:{X} {user_context or '无'}")
        print(f"  {D}Thread:{X} {state['thread_id']}")
        print(f"  {D}模式:{X} {'自动' if auto else '交互式'}")

        start_time = time.time()
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        last_trace_count = 0
        _no_progress_since = 0.0
        _think_notified = False

        NODE_DESCRIPTIONS = {
            "input_validator": "验证输入文件",
            "parser": "解析XML电检报告",
            "dtc_focus": "按优先级筛选重点DTC",
            "hitl_parse_confirm": "解析确认",
            "surveyor": "构建ECU拓扑图",
            "mcp_librarian": "检索知识库",
            "rca_engine": "ReAct循环根因分析",
            "hitl_low_confidence": "低置信度处理",
            "auditor": "逻辑审计检查",
            "hitl_audit_conflict": "审计冲突处理",
            "hitl_report_approval": "报告审批",
            "report_generator": "生成Markdown诊断报告",
        }

        async def _on_waiting():
            nonlocal last_trace_count, _no_progress_since, _think_notified
            try:
                snapshot = await g.graph.aget_state(run_config)
                current = snapshot.values
                if current:
                    traces = current.get("operator_trace", [])
                    new_items = traces[last_trace_count:]
                    if new_items:
                        if _think_notified:
                            print()
                        for item in new_items:
                            rca_stream(item, item.get("iteration", "?"))
                        last_trace_count = len(traces)
                        _no_progress_since = 0.0
                        _think_notified = False
                    elif _no_progress_since == 0.0:
                        _no_progress_since = time.time()
                    elif not _think_notified and time.time() - _no_progress_since > 5:
                        print(f"    {C}[LLM生成思考中，请耐心等待...]{X}")
                        _think_notified = True

                    conf_hist = current.get("confidence_history", [])
                    if conf_hist:
                        latest = conf_hist[-1]
                        conf = latest.get("confidence", 0)
                        closure = "✓" if latest.get("closure_passed") else "✗"
                        print(f"    {C}[进度]{X} Iter {latest.get('iteration', '?')}, 置信度: {conf:.0%}, 闭环保: {closure}")
            except Exception:
                pass

        async def _on_node_event(node_name, node_output):
            desc = NODE_DESCRIPTIONS.get(node_name, node_name)
            nd(node_name, desc)

            if node_name == "input_validator":
                if node_output.get("exit_code") == 0:
                    ok(node_name, "ok")
                else:
                    ok(node_name, "fail", node_output.get("error_message"))
            elif node_name == "parser":
                if node_output.get("parse_success"):
                    s = node_output.get("structured_data", {})
                    ok(node_name, "ok", f"{len(s.get('ecu_states', []))} ECU, {len(s.get('dtc_codes', []))} DTC")
                else:
                    ok(node_name, "fail", node_output.get("parse_error"))
            elif node_name == "dtc_focus":
                focused = node_output.get("focused_dtcs", [])
                deferred = node_output.get("deferred_dtcs", [])
                ok(node_name, "ok", f"{len(focused)} 重点, {len(deferred)} 延迟")
            elif node_name == "surveyor":
                topo = node_output.get("topology_graph", {})
                if topo:
                    ok(node_name, "ok", f"{len(topo.get('nodes', []))} 节点, {len(topo.get('edges', []))} 边")
                else:
                    ok(node_name, "空", "无拓扑数据")
            elif node_name == "mcp_librarian":
                rag = node_output.get("rag_knowledge", {})
                if hasattr(rag, 'dtc_definitions'):
                    defs = rag.dtc_definitions or []
                elif isinstance(rag, dict):
                    defs = rag.get("dtc_definitions", [])
                else:
                    defs = []
                ok(node_name, "ok", f"{len(defs)} DTC定义")
            elif node_name == "rca_engine":
                if node_output.get("error_message"):
                    ok(node_name, "fail", node_output.get("error_message"))
                else:
                    ok(node_name, "ok", f"{len(node_output.get('analysis_steps', []))} 步, 置信度 {node_output.get('confidence_score', 0):.0%}")
            elif node_name == "auditor":
                passed = node_output.get("audit_passed", False)
                ok(node_name, "passed" if passed else "conflict")
                for item in node_output.get("audit_log", []):
                    prefix = G + "✓" + X if "信息" in item or "通过" in item else Y + "!" + X
                    print(f"  {prefix} {item}")
                for c in node_output.get("conflicts", []):
                    print(f"  {R}[冲突]{X} {c}")
            elif node_name == "report_generator":
                ok(node_name, "ok")
            elif node_name.startswith("hitl_"):
                pass

        async def _on_hitl_display(node_name, question, options):
            print(f"\n{Y}{'⚡' * 30}{X}")
            print(f"{D}[HITL]{X} 节点: {node_name}")
            print(f"{D}问题:{X} {question}")
            print(f"{D}选项:{X}")
            for opt in options:
                print(f"  {opt}")
            print(f"{Y}{'⚡' * 30}{X}")

        async def _on_hitl_input(node_name, question, options):
            valid_tokens = set()
            for opt in options:
                token = str(opt).split(":", 1)[0].strip().upper()
                if token and len(token) <= 2:
                    valid_tokens.add(token)

            while True:
                raw = input("\n请选择: ").strip().upper()
                if not raw:
                    print(f"{R}错误: 请输入一个有效选项，不能留空{X}")
                    continue
                if raw in valid_tokens:
                    choice = raw
                    break
                if len(raw) == 1:
                    matching = [t for t in valid_tokens if t.startswith(raw)]
                    if matching:
                        choice = matching[0]
                        break
                matching_full = [opt for opt in options if str(opt).upper().startswith(raw)]
                if matching_full:
                    token = str(matching_full[0]).split(":", 1)[0].strip().upper()
                    if token and token in valid_tokens:
                        choice = token
                        break
                print(f"{R}无效选择，请从以下选项中选择: {', '.join(str(o) for o in options)}{X}")

            user_text = None
            if choice == "C":
                user_text = input("请输入补充信息: ").strip()
            elif choice == "B" and node_name == "hitl_report_approval":
                user_text = input("请输入修订要求: ").strip()
            else:
                selected_opt_text = ""
                for opt in options:
                    if str(opt).split(":", 1)[0].strip().upper() == choice:
                        selected_opt_text = str(opt)
                        break
                if "其他" in selected_opt_text:
                    user_text = input("请描述补充原因: ").strip()

            print(f"  {G}[选择]{X} {choice}" + (f" | {Y}输入:{X} {user_text}" if user_text else ""))
            return choice, user_text

        final_state = await g.run(state, interactive=(not auto),
                                   on_waiting=_on_waiting,
                                   on_node_event=_on_node_event,
                                   on_hitl_display=_on_hitl_display,
                                   on_hitl_input=_on_hitl_input)

        elapsed = time.time() - start_time

        report = final_state.get("final_report") or {}
        print_final_report(report)
        print(f"\n{D}总耗时:{X} {elapsed:.1f}秒")

        report_path = Path("logs/interactive_diagnostic_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(dict(final_state) if hasattr(final_state, '__iter__') else final_state, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n{G}[保存]{X} 完整报告: {report_path.absolute()}")

        markdown_report = final_state.get("report_markdown")
        if markdown_report:
            md_path = Path("logs/diagnostic_report.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_report)
            print(f"{G}[保存]{X} Markdown报告: {md_path.absolute()}")

        return final_state


def print_final_report(report):
    print(f"\n{G}{'=' * 70}{X}")
    print(f"{D}[最终诊断报告]{X}")
    print(f"{G}{'=' * 70}{X}")
    print(f"  {D}根因:{X}")
    print(f"  {report.get('root_cause', 'N/A')}")
    print(f"\n  {D}置信度:{X} {report.get('confidence_score', 'N/A')}")
    ecus = report.get('affected_ecus', [])
    print(f"\n  {D}受影响ECU:{X} {', '.join(str(e) for e in ecus[:20])}" + (f" ...等{len(ecus)}个" if len(ecus) > 20 else ""))
    recs = report.get('recommendations', [])
    print(f"\n  {D}维修建议:{X}")
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. {rec}")
    risks = report.get('risk_warnings', [])
    if risks:
        print(f"\n  {D}风险提示:{X}")
        for risk in risks:
            print(f"  ⚠ {risk}")
    print(f"{G}{'=' * 70}{X}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="交互式车辆故障诊断工具")
    parser.add_argument("xml_file", nargs="?", default="testing/test1.xml", help="电检XML报告路径")
    parser.add_argument("--context", "-c", default="", help="用户描述的故障现象")
    parser.add_argument("--auto", "-a", action="store_true", help="自动模式（不等待人工交互）")
    args = parser.parse_args()

    xml_path = Path(args.xml_file)
    if not xml_path.is_absolute():
        xml_path = _PROJECT_ROOT / xml_path
    if not xml_path.exists():
        print(f"{R}[错误]{X} 文件不存在: {xml_path}")
        sys.exit(1)

    os.chdir(_PROJECT_ROOT)
    asyncio.run(run_interactive(str(xml_path), args.context, auto=args.auto))
