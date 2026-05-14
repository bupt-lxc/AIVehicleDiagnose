import pytest
from unittest.mock import MagicMock, patch

from app.diagnostic.graph import AutoDiagnoseGraph
from app.diagnostic.state import DiagnosticState, create_initial_state

class TestAutoDiagnoseGraphRouting:
    @pytest.fixture
    def graph_instance(self):
        # Mock dependencies to avoid actual initialization of RAG/LLM etc.
        with patch('app.diagnostic.graph.RAGClient'), \
             patch('app.diagnostic.graph.OpenAICompatibleProvider'), \
             patch('app.diagnostic.graph.PromptRegistry'), \
             patch('app.diagnostic.graph.ToolRegistry'), \
             patch('os.makedirs'), \
             patch('sqlite3.connect'):
             
             graph = AutoDiagnoseGraph(
                 parser_agent=MagicMock(),
                 surveyor_agent=MagicMock(),
                 mcp_librarian=MagicMock(),
                 rca_engine=MagicMock(),
                 auditor_agent=MagicMock(),
                 report_generator=MagicMock(),
                 hitl_handler=MagicMock(),
                 db_path=":memory:"
             )
             return graph

    def test_route_after_validation_success(self, graph_instance):
        """测试验证成功后路由到parser"""
        state = create_initial_state("test input")
        state["exit_code"] = 0
        
        result = graph_instance._route_after_validation(state)
        assert result == "parser"

    def test_route_after_validation_failure(self, graph_instance):
        """测试验证失败后路由到end"""
        state = create_initial_state("test input")
        state["exit_code"] = 1
        
        result = graph_instance._route_after_validation(state)
        assert result == "end"

    def test_route_after_parser_success(self, graph_instance):
        """测试解析成功后路由到dtc_focus"""
        state = create_initial_state("test input")
        state["parse_success"] = True
        
        result = graph_instance._route_after_parser(state)
        assert result == "focus"

    def test_route_after_focus_hitl(self, graph_instance):
        state = create_initial_state("test input")
        state["hitl_required"] = True
        assert graph_instance._route_after_focus(state) == "hitl"

    def test_route_after_focus_surveyor(self, graph_instance):
        state = create_initial_state("test input")
        state["hitl_required"] = False
        assert graph_instance._route_after_focus(state) == "surveyor"

    def test_route_after_parser_retry(self, graph_instance):
        """测试解析失败且重试次数未超限时路由到retry"""
        state = create_initial_state("test input")
        state["parse_success"] = False
        state["retry_state"]["parser_retries"] = graph_instance.MAX_PARSER_RETRIES - 1
        
        result = graph_instance._route_after_parser(state)
        assert result == "retry"

    def test_route_after_parser_fallback(self, graph_instance):
        """测试解析失败且重试次数超限时路由到fallback"""
        state = create_initial_state("test input")
        state["parse_success"] = False
        state["retry_state"]["parser_retries"] = graph_instance.MAX_PARSER_RETRIES
        
        result = graph_instance._route_after_parser(state)
        assert result == "fallback"

    def test_route_after_fallback_success(self, graph_instance):
        """测试后备解析成功后路由到hitl"""
        state = create_initial_state("test input")
        state["parse_success"] = True
        
        result = graph_instance._route_after_fallback(state)
        assert result == "hitl"

    def test_route_after_fallback_has_dtc(self, graph_instance):
        """测试后备解析有结构化数据时路由到hitl"""
        state = create_initial_state("test input")
        state["parse_success"] = False
        state["structured_data"]["dtc_codes"] = [{"code": "P0135"}]
        
        result = graph_instance._route_after_fallback(state)
        assert result == "hitl"

    def test_route_after_fallback_error(self, graph_instance):
        """测试后备解析失败时路由到error"""
        state = create_initial_state("test input")
        state["parse_success"] = False
        
        result = graph_instance._route_after_fallback(state)
        assert result == "error"

    def test_route_after_rca_low_confidence(self, graph_instance):
        """测试RCA置信度低时路由到hitl"""
        state = create_initial_state("test input")
        state["confidence_score"] = graph_instance.MIN_CONFIDENCE - 0.1
        
        result = graph_instance._route_after_rca(state)
        assert result == "hitl"

    def test_route_after_rca_nonrecoverable_error(self, graph_instance):
        state = create_initial_state("test input")
        state["current_node"] = "rca_engine"
        state["error_message"] = "fatal"
        state["error_recoverable"] = False
        assert graph_instance._route_after_rca(state) == "error"

    def test_route_after_rca_high_confidence(self, graph_instance):
        """测试RCA置信度高时路由到auditor"""
        state = create_initial_state("test input")
        state["confidence_score"] = graph_instance.MIN_CONFIDENCE + 0.1
        
        result = graph_instance._route_after_rca(state)
        assert result == "auditor"

    def test_route_after_low_confidence_retry(self, graph_instance):
        """测试低置信度人为要求重试时路由到retry"""
        state = create_initial_state("test input")
        state["hitl_feedbacks"] = [{
            "point": "low_confidence",
            "user_choice": "D"
        }]
        
        result = graph_instance._route_after_low_confidence(state)
        assert result == "retry"

    def test_route_after_low_confidence_auditor(self, graph_instance):
        """测试低置信度人为接受或忽略时路由到auditor"""
        state = create_initial_state("test input")
        state["hitl_feedbacks"] = [{
            "point": "low_confidence",
            "user_choice": "A"  # Proceed to auditor
        }]
        
        result = graph_instance._route_after_low_confidence(state)
        assert result == "auditor"

    def test_route_after_audit_passed(self, graph_instance):
        """测试审计通过后路由到approval"""
        state = create_initial_state("test input")
        state["audit_passed"] = True
        
        result = graph_instance._route_after_audit(state)
        assert result == "approval"

    def test_route_after_audit_failed(self, graph_instance):
        """测试审计未通过后路由到hitl"""
        state = create_initial_state("test input")
        state["audit_passed"] = False
        
        result = graph_instance._route_after_audit(state)
        assert result == "hitl"

    def test_route_after_audit_conflict_retry(self, graph_instance):
        """测试审计冲突人为要求重试时路由到retry"""
        state = create_initial_state("test input")
        state["hitl_feedbacks"] = [{
            "point": "audit_conflict",
            "user_choice": "B"
        }]
        
        result = graph_instance._route_after_audit_conflict(state)
        assert result == "retry"

    def test_route_after_audit_conflict_approval(self, graph_instance):
        """测试审计冲突人为接受时路由到approval"""
        state = create_initial_state("test input")
        state["hitl_feedbacks"] = [{
            "point": "audit_conflict",
            "user_choice": "A"
        }]
        
        result = graph_instance._route_after_audit_conflict(state)
        assert result == "approval"

    def test_route_after_approval_approved(self, graph_instance):
        """测试报告获批后路由到approved"""
        state = create_initial_state("test input")
        state["hitl_feedbacks"] = [{
            "point": "report_approval",
            "user_choice": "A"
        }]
        
        result = graph_instance._route_after_approval(state)
        assert result == "approved"

    def test_route_after_approval_revise(self, graph_instance):
        """测试报告需修改时路由到revise"""
        state = create_initial_state("test input")
        state["hitl_feedbacks"] = [{
            "point": "report_approval",
            "user_choice": "B"
        }]
        
        result = graph_instance._route_after_approval(state)
        assert result == "revise"

    def test_route_after_approval_more_info(self, graph_instance):
        """测试报告需更多信息时路由到more_info"""
        state = create_initial_state("test input")
        state["hitl_feedbacks"] = [{
            "point": "report_approval",
            "user_choice": "C"
        }]
        
        result = graph_instance._route_after_approval(state)
        assert result == "more_info"

    def test_input_validator_node(self, graph_instance):
        """测试验证节点执行"""
        state = create_initial_state("P0135 Engine error")
        result = graph_instance._input_validator_node(state)
        assert result["current_node"] == "input_validator"
        assert result["exit_code"] == 0

    def test_error_handler_node(self, graph_instance):
        """测试错误处理节点"""
        state = create_initial_state("test input")
        result = graph_instance._error_handler_node(state)
        assert result["current_node"] == "error_handler"
        assert result["exit_code"] == 1
