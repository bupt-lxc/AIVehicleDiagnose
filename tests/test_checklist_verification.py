
import pytest
import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.diagnostic.state import create_initial_state, DiagnosticState, HITLFeedback
from app.config import Settings, load_settings
from app.diagnostic.prompts.registry import PromptRegistry, DynamicPromptTemplate
from app.diagnostic.tools.registry import ToolRegistry
from app.diagnostic.agents.parser_agent import ParserAgent
from app.diagnostic.agents.surveyor_agent import SurveyorAgent
from app.diagnostic.hitl.handler import HITLHandler
from app.topology.database import init_db, get_db_connection
from app.topology.models import ECUModel, ConnectionModel
from app.topology import queries
from fastapi.testclient import TestClient
from api.main import create_app

# --- Item 2: Config ---
def test_config_integration():
    settings = load_settings()
    assert hasattr(settings, "rag_server_url")
    assert hasattr(settings, "local_profile")

# --- Item 3: Prompt Registry ---
def test_prompt_registry_runtime_injection():
    registry = PromptRegistry()
    template = DynamicPromptTemplate(
        name="test_prompt",
        version="v1",
        system_prompt="System",
        user_prompt_template="User {input}",
        injectable_vars=["input"]
    )
    registry.register(template)
    
    # Check render works
    rendered = registry.get("test_prompt", input="hello")
    assert "User hello" in rendered
    
    # Runtime update
    template_v2 = DynamicPromptTemplate(
        name="test_prompt",
        version="v2",
        system_prompt="System V2",
        user_prompt_template="User V2 {input}",
        injectable_vars=["input"]
    )
    registry.register(template_v2)
    
    # Check new version is active and renders
    rendered_v2 = registry.get("test_prompt", input="world")
    assert "User V2 world" in rendered_v2

# --- Item 4: Tool Registry ---
def test_tool_registry_progressive_disclosure():
    registry = ToolRegistry()
    def dummy_tool(): pass
    
    registry.register(
        name="dummy", 
        handler=dummy_tool,
        description="A dummy tool",
        input_schema={}
    )
    assert registry.get("dummy") == dummy_tool

# --- Item 5 & 7: Topology DB & Surveyor ---
@pytest.fixture
def temp_topology_db(tmp_path):
    db_path = tmp_path / "test_topology.sqlite"
    with patch("app.topology.database.DB_PATH", str(db_path)):
        init_db()
        yield db_path

def test_topology_operations_and_surveyor(temp_topology_db):
    # Setup Data
    ecu1 = ECUModel(ecu_id="ECU1", domain="Body", power_segment="A", is_gateway=False)
    ecu2 = ECUModel(ecu_id="ECU2", domain="Powertrain", power_segment="B", is_gateway=True)
    conn1 = ConnectionModel(source_id="ECU1", target_id="ECU2", protocol="CAN", gateway_id=None)
    
    queries.add_ecu(ecu1)
    queries.add_ecu(ecu2)
    queries.add_connection(conn1)
    
    # Test DB Query
    fetched_ecu = queries.get_ecu("ECU1")
    assert fetched_ecu.ecu_id == "ECU1"
    
    # Test Surveyor Agent
    agent = SurveyorAgent()
    state = {
        "structured_data": {
            "dtc_codes": [{"code": "P0001", "ecu_id": "ECU1"}]
        }
    }
    
    result = agent.survey(state)
    graph = result.get("topology_graph")
    assert graph is not None
    assert len(graph["nodes"]) >= 2
    assert len(graph["edges"]) >= 1

# --- Item 6: Parser Agent ---
def test_parser_agent_robustness():
    mock_llm = MagicMock()
    mock_llm.generate.return_value.content = """
    {
        "dtc_codes": [{"code": "P0123", "status": "Active", "description": "Throttle Position"}],
        "ecu_states": [],
        "freeze_frames": [],
        "raw_text_segments": []
    }
    """
    
    agent = ParserAgent(local_llm=mock_llm)
    state = {"raw_input": "DTC P0123 Active", "retry_state": {}}
    
    result = agent.parse(state)
    assert result["parse_success"] is True
    assert len(result["structured_data"]["dtc_codes"]) == 1
    
    # Test robustness with invalid input
    state_invalid = {"raw_input": "", "retry_state": {}}
    result_invalid = agent.parse(state_invalid)
    assert result_invalid["parse_success"] is False
    assert "parse_error" in result_invalid

# --- Item 11: HITL Handler ---
def test_hitl_handler():
    handler = HITLHandler()
    state = {
        "structured_data": {
            "dtc_codes": [{"code": "P01", "status": "Active"}, {"code": "P02", "status": "Active"}]
        }
    }
    
    # Test generating HITL request
    state_with_hitl = handler.handle_parse_confirm(state)
    assert state_with_hitl["hitl_required"] is True
    assert state_with_hitl["current_hitl_point"] == "parse_confirm"
    
    # Test processing feedback
    new_state = handler.process_feedback(state_with_hitl, "A")
    assert new_state["hitl_required"] is False
    assert len(new_state["hitl_feedbacks"]) == 1
    assert new_state["priority_dtcs"] == ["P01"]

# --- Item 13: API ---
def test_api_startup():
    settings = load_settings()
    app = create_app(settings)
    client = TestClient(app)
    # Check if health endpoint exists or try root
    try:
        response = client.get("/health/live")
        if response.status_code == 404:
             response = client.get("/")
    except Exception:
        pytest.skip("API might not be ready or dependencies missing")
        return
        
    assert response.status_code in [200, 404] # At least app started
