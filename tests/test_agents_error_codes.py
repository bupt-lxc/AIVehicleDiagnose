import pytest
from unittest.mock import MagicMock
from app.diagnostic.state import create_initial_state, DiagnosticState
from app.diagnostic.agents.rca_engine import RCAEngine, RCAConfig
from app.diagnostic.agents.parser_agent import ParserAgent
from app.diagnostic.agents.auditor_agent import AuditorAgent
from app.diagnostic.prompts.registry import PromptRegistry

class MockLLM:
    def generate(self, *args, **kwargs):
        return MagicMock(content="{}")

@pytest.fixture
def mock_state():
    return create_initial_state("test input")

def test_rca_engine_max_retries():
    # Setup RCA Engine
    llm = MagicMock()
    registry = MagicMock()
    tools = MagicMock()
    tools.list_openai_tools.return_value = []
    
    config = RCAConfig(max_retries=3)
    
    # Create engine
    engine = RCAEngine(llm, registry, tools, config)
    
    # Create state
    mock_state = create_initial_state("test input")
    mock_state["parse_success"] = True # Set to True to pass integrity check
    mock_state["structured_data"]["dtc_codes"] = [{"code": "P0123"}]
    mock_state["retry_state"]["rca_retries"] = 3
    
    result = engine.analyze(mock_state)
    
    assert result.get("exit_code") == 5
    assert "达到最大重试次数" in result.get("error_message", "")

def test_parser_agent_validation_failure(mock_state):
    agent = ParserAgent()
    
    # Empty input should fail validation
    mock_state["raw_input"] = ""
    
    result = agent.parse(mock_state)
    
    assert result["exit_code"] == 1
    assert "输入不能为空" in result["parse_error"]

def test_auditor_agent_failure(mock_state):
    agent = AuditorAgent()
    
    # Setup state to fail audit (e.g. no root cause)
    mock_state["preliminary_root_cause"] = ""
    mock_state["analysis_steps"] = ["step 1"]
    
    result = agent.audit(mock_state)
    
    assert result["audit_passed"] is False
    assert result["exit_code"] == 6
