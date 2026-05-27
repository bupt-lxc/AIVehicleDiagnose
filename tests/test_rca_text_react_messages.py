import unittest

from app.diagnostic.agents.rca_engine import RCAConfig, RCAEngine
from app.diagnostic.prompts.registry import DynamicPromptTemplate, PromptRegistry
from app.diagnostic.state import create_initial_state
from app.diagnostic.tools.registry import ToolRegistry


class _Response:
    def __init__(self, content, raw=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.raw = raw or {}


class _CapturingLLM:
    def __init__(self):
        self.calls = []

    def generate(self, messages, temperature=0.2, tools=None, tool_choice=None):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            return _Response('Thought: inspect DTC\nAction: query_dtc\nAction Input: {"dtc_code": "P0135", "limit": 1}')
        return _Response("Final Answer: 根因: ECU ECU1 oxygen sensor circuit fault. 置信度: 0.85")


class _DeepSeekCapturingLLM:
    def __init__(self):
        self.calls = []

    def generate(self, messages, temperature=0.2, tools=None, tool_choice=None):
        self.calls.append(list(messages))
        first_content = 'Thought: inspect DTC\nAction: query_dtc\nAction Input: {"dtc_code": "P0135", "limit": 1}'
        if len(self.calls) == 1:
            return _Response(
                first_content,
                raw={
                    "choices": [
                        {
                            "message": {
                                "content": first_content,
                                "reasoning_content": "DeepSeek thinking payload",
                            }
                        }
                    ]
                },
            )
        return _Response("Final Answer: root cause: ECU ECU1 oxygen sensor circuit fault. confidence: 0.85")


class _MultiToolCallCapturingLLM:
    def __init__(self):
        self.calls = []

    def generate(self, messages, temperature=0.2, tools=None, tool_choice=None):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            return _Response(
                "",
                tool_calls=[
                    {
                        "id": "call_query",
                        "type": "function",
                        "function": {
                            "name": "query_dtc",
                            "arguments": '{"dtc_code": "P0135", "limit": 1}',
                        },
                    },
                    {
                        "id": "call_hop",
                        "type": "function",
                        "function": {
                            "name": "check_ecu_hop_distance",
                            "arguments": '{"source_ecu": "ECU1", "target_ecu": "ECU2"}',
                        },
                    },
                ],
            )
        return _Response("Final Answer: root cause: ECU ECU1 sensor fault. confidence: 0.85")


class RCATextReactMessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_react_observation_is_sent_as_user_context(self):
        prompts = PromptRegistry()
        prompts.register(
            DynamicPromptTemplate(
                name="rca",
                version="v1",
                system_prompt="Use ReAct.",
                user_prompt_template="DTCs={dtc_codes}\nTopology={topology_graph}\nCases={case_studies}\nContext={user_context}",
            )
        )
        tools = ToolRegistry()
        tools.register(
            name="query_dtc",
            handler=lambda dtc_code, limit=1: {"success": True, "dtc_code": dtc_code, "ecu_id": "ECU1"},
            description="query dtc",
            input_schema={
                "type": "object",
                "properties": {"dtc_code": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["dtc_code"],
            },
        )
        llm = _CapturingLLM()
        engine = RCAEngine(
            llm=llm,
            prompt_registry=prompts,
            tool_registry=tools,
            config=RCAConfig(enable_progressive_disclosure=False, verbose=False),
        )
        state = create_initial_state("DTC P0135", "parking fault")
        state["parse_success"] = True
        state["structured_data"]["dtc_codes"] = [
            {"code": "P0135", "status": "Active", "description": "O2 Sensor", "ecu_id": "ECU1"}
        ]
        state["focused_dtcs"] = list(state["structured_data"]["dtc_codes"])
        state["topology_graph"] = {
            "nodes": [{"ecu_id": "ECU1", "domain": "Powertrain", "power_segment": "P1", "gateway": False}],
            "edges": [],
            "depth": 1,
            "center_ecu": "ECU1",
        }
        state["rag_knowledge"] = {
            "dtc_definitions": [{"code": "P0135", "definition": "O2 sensor circuit", "source": "test"}],
            "case_studies": [],
            "repair_procedures": [],
        }

        await engine.analyze(state)

        second_call_messages = llm.calls[1]
        self.assertEqual(second_call_messages[-1]["role"], "user")
        self.assertIn("Observation:", second_call_messages[-1]["content"])

    async def test_assistant_reasoning_content_is_sent_back_on_next_call(self):
        prompts = PromptRegistry()
        prompts.register(
            DynamicPromptTemplate(
                name="rca",
                version="v1",
                system_prompt="Use ReAct.",
                user_prompt_template="DTCs={dtc_codes}\nTopology={topology_graph}\nCases={case_studies}\nContext={user_context}",
            )
        )
        tools = ToolRegistry()
        tools.register(
            name="query_dtc",
            handler=lambda dtc_code, limit=1: {"success": True, "dtc_code": dtc_code, "ecu_id": "ECU1"},
            description="query dtc",
            input_schema={
                "type": "object",
                "properties": {"dtc_code": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["dtc_code"],
            },
        )
        llm = _DeepSeekCapturingLLM()
        engine = RCAEngine(
            llm=llm,
            prompt_registry=prompts,
            tool_registry=tools,
            config=RCAConfig(enable_progressive_disclosure=False, verbose=False),
        )
        state = create_initial_state("DTC P0135", "parking fault")
        state["parse_success"] = True
        state["structured_data"]["dtc_codes"] = [
            {"code": "P0135", "status": "Active", "description": "O2 Sensor", "ecu_id": "ECU1"}
        ]
        state["focused_dtcs"] = list(state["structured_data"]["dtc_codes"])
        state["topology_graph"] = {
            "nodes": [{"ecu_id": "ECU1", "domain": "Powertrain", "power_segment": "P1", "gateway": False}],
            "edges": [],
            "depth": 1,
            "center_ecu": "ECU1",
        }
        state["rag_knowledge"] = {
            "dtc_definitions": [{"code": "P0135", "definition": "O2 sensor circuit", "source": "test"}],
            "case_studies": [],
            "repair_procedures": [],
        }

        await engine.analyze(state)

        second_call_messages = llm.calls[1]
        assistant_messages = [item for item in second_call_messages if item["role"] == "assistant"]
        self.assertEqual(assistant_messages[-1]["reasoning_content"], "DeepSeek thinking payload")

    async def test_native_multi_tool_calls_all_receive_tool_messages(self):
        prompts = PromptRegistry()
        prompts.register(
            DynamicPromptTemplate(
                name="rca",
                version="v1",
                system_prompt="Use native tools.",
                user_prompt_template="DTCs={dtc_codes}\nTopology={topology_graph}\nCases={case_studies}\nContext={user_context}",
            )
        )
        tools = ToolRegistry()
        tools.register(
            name="query_dtc",
            handler=lambda dtc_code, limit=1: {"success": True, "dtc_code": dtc_code, "ecu_id": "ECU1"},
            description="query dtc",
            input_schema={
                "type": "object",
                "properties": {"dtc_code": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["dtc_code"],
            },
        )
        tools.register(
            name="check_ecu_hop_distance",
            handler=lambda source_ecu, target_ecu: {"success": True, "distance": 1},
            description="check ecu hop",
            input_schema={
                "type": "object",
                "properties": {"source_ecu": {"type": "string"}, "target_ecu": {"type": "string"}},
                "required": ["source_ecu", "target_ecu"],
            },
        )
        llm = _MultiToolCallCapturingLLM()
        engine = RCAEngine(
            llm=llm,
            prompt_registry=prompts,
            tool_registry=tools,
            config=RCAConfig(enable_progressive_disclosure=False, verbose=False),
        )
        state = create_initial_state("DTC P0135", "parking fault")
        state["parse_success"] = True
        state["structured_data"]["dtc_codes"] = [
            {"code": "P0135", "status": "Active", "description": "O2 Sensor", "ecu_id": "ECU1"}
        ]
        state["focused_dtcs"] = list(state["structured_data"]["dtc_codes"])
        state["topology_graph"] = {
            "nodes": [
                {"ecu_id": "ECU1", "domain": "Powertrain", "power_segment": "P1", "gateway": False},
                {"ecu_id": "ECU2", "domain": "Body", "power_segment": "P1", "gateway": False},
            ],
            "edges": [],
            "depth": 1,
            "center_ecu": "ECU1",
        }
        state["rag_knowledge"] = {
            "dtc_definitions": [{"code": "P0135", "definition": "O2 sensor circuit", "source": "test"}],
            "case_studies": [],
            "repair_procedures": [],
        }

        await engine.analyze(state)

        second_call_messages = llm.calls[1]
        tool_call_ids = [item.get("tool_call_id") for item in second_call_messages if item["role"] == "tool"]
        self.assertIn("call_query", tool_call_ids)
        self.assertIn("call_hop", tool_call_ids)


if __name__ == "__main__":
    unittest.main()
