import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from app.diagnostic.graph import AutoDiagnoseGraph
from app.diagnostic.state import create_initial_state
from app.diagnostic.hitl.handler import HITLHandler

@pytest.mark.asyncio
async def test_hitl_parse_confirm_flow():
    # 1. Setup Graph with in-memory DB
    db_path = ":memory:"
    
    # Mock Parser to return success with multiple DTCs to trigger HITL
    mock_parser = MagicMock()
    mock_parser.parse.return_value = {
        "parse_success": True,
        "structured_data": {
            "dtc_codes": [{"code": "P0123"}, {"code": "P0456"}]
        },
        "current_node": "parser"
    }
    
    # Mock other agents
    mock_surveyor = MagicMock()
    mock_surveyor.survey.return_value = {"current_node": "surveyor"}
    
    mock_mcp = AsyncMock()
    mock_mcp.retrieve.return_value = {"current_node": "mcp_librarian"}
    
    mock_rca = MagicMock()
    mock_rca.analyze.return_value = {
        "current_node": "rca_engine", 
        "confidence_score": 0.9,
        "preliminary_root_cause": "Test Cause",
        "analysis_steps": ["step1", "step2"],
        "parse_success": True, # Ensure integrity
        "structured_data": {"dtc_codes": [{"code": "P0123"}]}
    }
    
    mock_auditor = MagicMock()
    mock_auditor.audit.return_value = {
        "current_node": "auditor", 
        "audit_passed": True,
        "conflicts": []
    }
    
    mock_report = MagicMock()
    mock_report.generate.return_value = {
        "current_node": "report_generator", 
        "final_report": {"root_cause": "Test Cause"}
    }
    
    # Real HITL Handler
    hitl_handler = HITLHandler()
    
    graph_runner = AutoDiagnoseGraph(
        parser_agent=mock_parser,
        surveyor_agent=mock_surveyor,
        mcp_librarian=mock_mcp,
        rca_engine=mock_rca,
        auditor_agent=mock_auditor,
        report_generator=mock_report,
        hitl_handler=hitl_handler,
        db_path=db_path
    )
    
    async with graph_runner:
        # 2. Run initial analysis
        initial_state = create_initial_state("test input", thread_id="test_thread")
        
        # Run until interrupt (after hitl_parse_confirm)
        state_after_run = await graph_runner.run(initial_state)
        
        # Verify it stopped at hitl_parse_confirm
        assert state_after_run["hitl_required"] is True
        assert state_after_run["current_hitl_point"] == "parse_confirm"
        
        # 3. Simulate User Feedback (API Logic) - Select first DTC
        config = {"configurable": {"thread_id": "test_thread"}}
        current_snapshot = await graph_runner.graph.aget_state(config)
        current_state = current_snapshot.values
        
        # User selects 'A' (first option)
        updated_state = hitl_handler.process_feedback(current_state, "A")
        
        # Verify update logic
        assert updated_state["hitl_required"] is False
        assert updated_state["priority_dtcs"] == ["P0123"]
        
        # Update graph state
        await graph_runner.graph.aupdate_state(config, updated_state)
        
        # 4. Resume
        state_at_approval = await graph_runner.graph.ainvoke(None, config)
        
        assert state_at_approval["hitl_required"] is True
        assert state_at_approval["current_hitl_point"] == "report_approval"
        
        # 5. Approve Report
        current_snapshot = await graph_runner.graph.aget_state(config)
        updated_state = hitl_handler.process_feedback(current_snapshot.values, "A") # A = Approve
        await graph_runner.graph.aupdate_state(config, updated_state)
        
        # Resume to finish
        final_state = await graph_runner.graph.ainvoke(None, config)
        
        assert final_state["report_approved"] is True
        assert final_state["final_report"]["root_cause"] == "Test Cause"
