import asyncio
from unittest.mock import MagicMock, patch

from app.diagnostic.graph import AutoDiagnoseGraph
from app.diagnostic.hitl.handler import HITLHandler
from app.diagnostic.prioritizer import DTCFocusEngine
from app.diagnostic.state import create_initial_state
from app.diagnostic.tools.registry import ToolRegistry, ProgressiveDisclosureController


def test_dtc_focus_engine_handles_large_pool():
    engine = DTCFocusEngine(max_focused=15)
    dtcs = []
    for idx in range(250):
        dtcs.append(
            {
                "code": f"U{idx:04d}" if idx % 10 == 0 else f"P{idx:04d}",
                "severity": "High" if idx % 7 == 0 else "Medium",
                "description": "network communication failure" if idx % 10 == 0 else "sensor drift",
                "priority_score": idx % 100,
                "ecu_id": f"ECU{idx % 12}",
            }
        )
    result = engine.focus(dtcs=dtcs, user_context="network communication", preferred_ecus=["ECU1"])
    assert len(result["all_dtcs"]) == 250
    assert len(result["focused_dtcs"]) == 15
    assert len(result["deferred_dtcs"]) == 235
    assert len(result["dtc_selection_log"]) == 250
    assert all("focus_score" in item for item in result["focused_dtcs"])


def test_progressive_disclosure_prefers_exploration_on_low_confidence():
    registry = ToolRegistry()
    registry.register(
        name="t0",
        handler=lambda **_: {"ok": True},
        description="t0",
        input_schema={"type": "object", "properties": {}},
        disclosure_level=0,
    )
    registry.register(
        name="t3",
        handler=lambda **_: {"ok": True},
        description="t3",
        input_schema={"type": "object", "properties": {}},
        disclosure_level=3,
    )
    ctl = ProgressiveDisclosureController(registry)
    assert ctl.update_level(0.2, {}) == 3
    assert ctl.update_level(0.9, {}) == 0


def test_hitl_focus_adjustment_can_override_focus_pool():
    handler = HITLHandler()
    state = create_initial_state("raw", "ctx")
    state["current_hitl_point"] = "focus_adjustment"
    state["hitl_options"] = ["A", "D"]
    state["all_dtcs"] = [
        {"code": "P0100", "ecu_id": "ECU_A", "status": "Active", "description": "", "severity": "High", "priority_score": 80},
        {"code": "U0101", "ecu_id": "ECU_B", "status": "Stored", "description": "", "severity": "Medium", "priority_score": 40},
    ]
    state["focused_dtcs"] = list(state["all_dtcs"])
    state["structured_data"]["dtc_codes"] = list(state["all_dtcs"])
    updated = handler.process_feedback(state, "A", "ECU_B")
    assert len(updated["focused_dtcs"]) == 1
    assert updated["focused_dtcs"][0]["ecu_id"] == "ECU_B"
    assert len(updated["deferred_dtcs"]) == 1


def test_graph_nodes_append_transition_trace():
    with patch("app.diagnostic.graph.RAGClient"), patch("app.diagnostic.graph.OpenAICompatibleProvider"), patch("app.diagnostic.graph.PromptRegistry"), patch("app.diagnostic.graph.ToolRegistry"):
        graph = AutoDiagnoseGraph(
            parser_agent=MagicMock(),
            surveyor_agent=MagicMock(),
            mcp_librarian=MagicMock(),
            rca_engine=MagicMock(),
            auditor_agent=MagicMock(),
            report_generator=MagicMock(),
            hitl_handler=MagicMock(),
            db_path=":memory:",
        )
    state = create_initial_state("raw")
    out = graph._error_handler_node(state)
    assert out["transition_log"]
    assert out["operator_trace"]
    assert out["operator_trace"][-1]["type"] == "transition"


def test_degraded_mode_can_be_set_without_online_key():
    with patch("app.diagnostic.graph.load_settings") as mocked:
        settings = MagicMock()
        settings.online_profile.api_key = ""
        mocked.return_value = settings
        with patch("app.diagnostic.graph.RAGClient"), patch("app.diagnostic.graph.OpenAICompatibleProvider"), patch("app.diagnostic.graph.PromptRegistry"), patch("app.diagnostic.graph.ToolRegistry"):
            graph = AutoDiagnoseGraph(
                parser_agent=MagicMock(),
                surveyor_agent=MagicMock(),
                mcp_librarian=MagicMock(),
                rca_engine=MagicMock(),
                auditor_agent=MagicMock(),
                report_generator=MagicMock(),
                hitl_handler=MagicMock(),
                db_path=":memory:",
            )
        state = create_initial_state("raw")
        graph.graph = MagicMock()
        async def fake_invoke(current_state, _):
            return current_state
        graph.graph.ainvoke = fake_invoke
        result = asyncio.run(graph.run(state, {"configurable": {"thread_id": state["thread_id"]}}))
        assert result["degraded_mode"] is True
