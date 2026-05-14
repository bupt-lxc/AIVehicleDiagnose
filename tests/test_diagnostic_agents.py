import pytest
from app.diagnostic.state import create_initial_state
from app.diagnostic.agents import RCAEngine, RCAConfig, AuditorAgent, ReportGenerator
from app.diagnostic.prompts.registry import PromptRegistry, DynamicPromptTemplate
from app.diagnostic.tools.registry import ToolRegistry
from unittest.mock import MagicMock

class MockLLMResponse:
    def __init__(self, content):
        self.content = content

class MockLLMProvider:
    def __init__(self, name="mock_llm", response_content="Mock response"):
        self.name = name
        self.response_content = response_content

    def generate(self, messages, temperature=0.2, tools=None, tool_choice=None):
        return MockLLMResponse(self.response_content)

@pytest.fixture
def mock_state():
    state = create_initial_state(raw_input="DTC: P0135 Active")
    state["structured_data"]["dtc_codes"] = [{"code": "P0135", "status": "Active", "description": "O2 Sensor", "ecu_id": "ECU_PTC_001"}]
    return state

@pytest.fixture
def prompt_registry():
    registry = PromptRegistry()
    registry.register(DynamicPromptTemplate(
        name="rca",
        version="v1",
        system_prompt="System Prompt",
        user_prompt_template="User Prompt",
    ))
    registry.register(DynamicPromptTemplate(
        name="report",
        version="v1",
        system_prompt="Report System Prompt",
        user_prompt_template="Report User Prompt",
    ))
    return registry

@pytest.fixture
def tool_registry():
    return ToolRegistry()

def test_rca_engine(mock_state, prompt_registry, tool_registry):
    llm = MockLLMProvider(response_content="Thought: I should check things.\nFinal Answer: root cause: Faulty sensor\nconfidence: 0.85")
    config = RCAConfig(enable_progressive_disclosure=False)
    engine = RCAEngine(llm=llm, prompt_registry=prompt_registry, tool_registry=tool_registry, config=config)
    
    # State check requires run_id, thread_id, etc. which are in create_initial_state
    mock_state["parse_success"] = True
    
    result = engine.analyze(mock_state)
    assert result["current_node"] == "rca_engine"
    assert result["confidence_score"] == 0.85
    assert result["preliminary_root_cause"] == "Faulty sensor"

def test_auditor_agent(mock_state):
    agent = AuditorAgent()
    mock_state["analysis_steps"] = ["Step 1", "Step 2", "Step 3: Analyze reason for P0135"]
    mock_state["preliminary_root_cause"] = "Faulty sensor P0135"
    mock_state["confidence_score"] = 0.85
    
    result = agent.audit(mock_state)
    assert result["current_node"] == "auditor"
    assert result["audit_passed"] is True
    assert len(result["conflicts"]) == 0

def test_report_generator(mock_state, prompt_registry):
    llm = MockLLMProvider(response_content="# 车辆故障诊断报告\n## 维修建议\n- Replace sensor\n## 风险提示\n- High voltage")
    generator = ReportGenerator(llm=llm, prompt_registry=prompt_registry)
    
    mock_state["preliminary_root_cause"] = "Faulty sensor"
    mock_state["confidence_score"] = 0.85
    
    result = generator.generate(mock_state)
    assert result["current_node"] == "report_generator"
    assert result["exit_code"] == 0
    assert result["final_report"]["root_cause"] == "Faulty sensor"
    assert result["final_report"]["recommendations"] == ["Replace sensor"]
    assert result["final_report"]["risks"] == ["High voltage"]
