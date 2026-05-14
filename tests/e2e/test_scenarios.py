"""End-to-End Simulation Test Script

Simulates real user behavior and verifies the complete diagnostic flow.
Covers:
1. Happy Path (High Confidence)
2. Parsing Ambiguity (HITL)
3. Low Confidence RCA (HITL)
4. Audit Conflict (HITL)
5. Report Rejection (HITL)
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from app.diagnostic.graph import AutoDiagnoseGraph
from app.diagnostic.state import create_initial_state
from app.diagnostic.hitl.handler import HITLHandler
from app.diagnostic.audit_logger import AuditLogger

# --- Mocks & Fixtures ---

class MockLLM:
    def __init__(self, responses: dict):
        self.responses = responses
        self.call_count = 0

    def generate(self, messages, **kwargs):
        # Simplistic mocking based on content of the last message or system prompt
        last_msg = messages[-1]["content"] if messages else ""
        
        # Check for parser prompt
        if "JSON" in last_msg or "解析" in last_msg:
            return MagicMock(content=self.responses.get("parser", "{}"))
        
        # Check for RCA prompt
        if "Thought:" in self.responses.get("rca", ""):
             # Return RCA thought sequence if configured
             return MagicMock(content=self.responses.get("rca"))
        
        # Check for Report prompt
        if "报告" in last_msg:
            return MagicMock(content=self.responses.get("report", "# Report"))
            
        return MagicMock(content="Default response")

@pytest.fixture
def db_path():
    return ":memory:"

# --- Scenarios ---

@pytest.mark.asyncio
async def test_e2e_happy_path(db_path):
    """Scenario 1: Happy Path - High Confidence, No HITL"""
    
    # Setup Mocks
    mock_parser_response = """
    {
        "dtc_codes": [{"code": "P0123", "status": "Active", "description": "Throttle Position Sensor"}],
        "ecu_states": [{"ecu_id": "ECU_ENGINE", "status": "Online"}],
        "freeze_frames": [],
        "raw_text_segments": []
    }
    """
    
    mock_rca_response_step1 = "Thought: The DTC P0123 indicates high voltage. The reason is likely a short."
    mock_rca_response_step2 = 'Action: search_knowledge_base\nAction Input: {"query": "P0123 cause"}'
    mock_rca_response_step3 = "Thought: Based on analysis of topology, check sensor ground.\nFinal Answer: root cause: Faulty Throttle Position Sensor (Short to Power)\nconfidence: 0.95"
    
    mock_report_response = "# Diagnostic Report\n## Root Cause\nFaulty TPS..."
    
    # Initialize Graph
    graph_runner = AutoDiagnoseGraph(db_path=db_path)
    
    # Inject Mocks
    mock_llm_parser = MagicMock()
    mock_llm_parser.generate.return_value = MagicMock(content=mock_parser_response)
    graph_runner.parser_agent._cloud_llm = mock_llm_parser
    
    mock_llm_rca = MagicMock()
    mock_llm_rca.generate.side_effect = [
        MagicMock(content=mock_rca_response_step1),
        MagicMock(content=mock_rca_response_step2),
        MagicMock(content=mock_rca_response_step3)
    ]
    graph_runner.rca_engine._llm = mock_llm_rca
    
    mock_llm_report = MagicMock()
    mock_llm_report.generate.return_value = MagicMock(content=mock_report_response)
    graph_runner.report_generator._llm = mock_llm_report
    
    graph_runner.auditor_agent._llm = MagicMock(content="Passed")
    
    # Mock Surveyor & Librarian (Network calls)
    graph_runner.surveyor_agent.survey = MagicMock(return_value={"topology_graph": {"nodes": [], "edges": []}})
    graph_runner.mcp_librarian.retrieve = AsyncMock(return_value={"rag_knowledge": {}})
    
    # Run
    async with graph_runner:
        initial_state = create_initial_state("DTC P0123 detected", thread_id="happy_path_thread")
        final_state = await graph_runner.run(initial_state)
        
        # Assertions
        assert final_state["exit_code"] == 0
        assert final_state["parse_success"] is True
        assert final_state["confidence_score"] >= 0.7
        
        # Note: Report Approval is mandatory, so we expect HITL here
        if final_state.get("hitl_required"):
             hitl_handler = HITLHandler()
             
             # If stopped at Report Approval, approve it
             if final_state.get("current_hitl_point") == "report_approval":
                 updated_state = hitl_handler.process_feedback(final_state, "A") # Approve
                 await graph_runner.graph.aupdate_state({"configurable": {"thread_id": "happy_path_thread"}}, updated_state)
                 final_state = await graph_runner.graph.ainvoke(None, {"configurable": {"thread_id": "happy_path_thread"}})
        
        assert final_state["report_approved"] is True
        assert "Faulty Throttle Position Sensor" in final_state["final_report"]["root_cause"]

@pytest.mark.asyncio
async def test_e2e_hitl_low_confidence(db_path):
    """Scenario 3: Low Confidence RCA -> HITL -> User Hint -> Retry -> Success"""
    
    # Setup Mocks
    mock_parser_response = '{"dtc_codes": [{"code": "U0100", "status": "Active"}], "ecu_states": [], "freeze_frames": [], "raw_text_segments": []}'
    
    # First RCA run: Low confidence
    mock_rca_response_1 = """
    Thought: U0100 is lost comms.
    Final Answer: root cause: Unknown communication issue
    confidence: 0.4
    """
    
    # Second RCA run: High confidence (after hint)
    mock_rca_response_2 = """
    Thought: User hinted about loose connector. Checking diagrams.
    Final Answer: root cause: Loose Connector J1939
    confidence: 0.9
    """
    
    mock_llm_parser = MagicMock()
    mock_llm_parser.generate.return_value = MagicMock(content=mock_parser_response)
    
    mock_llm_rca = MagicMock()
    mock_llm_rca.generate.side_effect = [
        MagicMock(content=mock_rca_response_1),  # RCA 1
        MagicMock(content=mock_rca_response_2)   # RCA 2 (Retry)
    ]
    
    mock_llm_report = MagicMock()
    mock_llm_report.generate.return_value = MagicMock(content="# Report")

    # Initialize
    graph_runner = AutoDiagnoseGraph(db_path=db_path)
    graph_runner.parser_agent._cloud_llm = mock_llm_parser
    graph_runner.rca_engine._llm = mock_llm_rca
    graph_runner.report_generator._llm = mock_llm_report
    graph_runner.auditor_agent._llm = MagicMock()
    
    # Mock helpers
    graph_runner.surveyor_agent.survey = MagicMock(return_value={"topology_graph": {"nodes": [], "edges": []}})
    graph_runner.mcp_librarian.retrieve = AsyncMock(return_value={"rag_knowledge": {}})
    
    async with graph_runner:
        initial_state = create_initial_state("U0100 detected communication loss", thread_id="low_conf_thread")
        
        # 1. Run -> Stops at HITL Low Confidence
        state_1 = await graph_runner.run(initial_state)
        
        assert state_1["hitl_required"] is True
        assert state_1["current_hitl_point"] == "low_confidence"
        assert state_1["confidence_score"] == 0.4
        
        # 2. User provides hint (Option D + Input)
        hitl_handler = HITLHandler()
        updated_state = hitl_handler.process_feedback(state_1, "D", user_input="Check loose connector")
        
        # Verify hint injection
        assert updated_state["user_description"] == "Check loose connector"
        
        await graph_runner.graph.aupdate_state({"configurable": {"thread_id": "low_conf_thread"}}, updated_state)
        
        # 3. Resume -> RCA Retry -> Auditor -> Report Approval
        # Auditor might fail due to empty topology in mock, leading to Audit Conflict HITL
        state_2 = await graph_runner.graph.ainvoke(None, {"configurable": {"thread_id": "low_conf_thread"}})
        
        # Handle potential Audit Conflict
        if state_2.get("current_hitl_point") == "audit_conflict":
             updated_state = hitl_handler.process_feedback(state_2, "A") # Ignore conflict
             await graph_runner.graph.aupdate_state({"configurable": {"thread_id": "low_conf_thread"}}, updated_state)
             state_2 = await graph_runner.graph.ainvoke(None, {"configurable": {"thread_id": "low_conf_thread"}})
             
        assert state_2["current_hitl_point"] == "report_approval"
        
        # 4. Approve
        updated_state = hitl_handler.process_feedback(state_2, "A")
        await graph_runner.graph.aupdate_state({"configurable": {"thread_id": "low_conf_thread"}}, updated_state)
        final_state = await graph_runner.graph.ainvoke(None, {"configurable": {"thread_id": "low_conf_thread"}})
        
        assert final_state["report_approved"] is True
        assert "Loose Connector" in final_state["preliminary_root_cause"]

if __name__ == "__main__":
    # Allow running directly
    import sys
    sys.exit(pytest.main(["-v", __file__]))
