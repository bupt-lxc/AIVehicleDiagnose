import pytest
from app.diagnostic.validators import (
    InputValidator,
    StructuredDataValidator,
    StateTransitionValidator,
    ValidationResult,
)

class TestInputValidator:
    @pytest.fixture
    def validator(self):
        return InputValidator()

    def test_empty_input(self, validator):
        """测试空输入验证"""
        result = validator.validate_raw_input("")
        assert not result.is_valid
        assert "输入不能为空" in result.errors
        assert result.sanitized_data is None

    def test_short_input(self, validator):
        """测试输入长度不足"""
        result = validator.validate_raw_input("123")
        assert not result.is_valid
        assert any("输入长度不足" in err for err in result.errors)

    def test_long_input(self, validator):
        """测试输入长度超限"""
        long_input = "A" * (validator.RAW_INPUT_MAX_LENGTH + 1)
        result = validator.validate_raw_input(long_input)
        assert not result.is_valid
        assert any("输入长度超限" in err for err in result.errors)

    def test_valid_input_with_dtc(self, validator):
        """测试包含DTC的有效输入"""
        input_text = "车辆出现故障码 P0135，发动机灯亮起。"
        result = validator.validate_raw_input(input_text)
        assert result.is_valid
        assert not result.errors
        assert not result.warnings
        assert result.sanitized_data == input_text

    def test_valid_input_without_valid_content(self, validator):
        """测试没有有效诊断信息的输入"""
        input_text = "今天天气很好，我开着车出去玩。"
        result = validator.validate_raw_input(input_text)
        assert result.is_valid  # 只要长度合适，算valid
        assert not result.errors
        assert any("可能不包含有效的诊断信息" in warn for warn in result.warnings)

    def test_sanitize_input(self, validator):
        """测试输入清理，去除控制字符"""
        input_text = "故障码\x00P0135\x1f"
        result = validator.validate_raw_input(input_text)
        assert result.sanitized_data == "故障码P0135"


class TestStructuredDataValidator:
    @pytest.fixture
    def validator(self):
        return StructuredDataValidator()

    def test_empty_data(self, validator):
        """测试空数据"""
        result = validator.validate({})
        assert not result.is_valid
        assert "结构化数据为空，无法进行诊断" in result.errors

    def test_valid_dtc(self, validator):
        """测试有效的DTC"""
        data = {
            "dtc_codes": [{"code": "P0135", "status": "Active"}]
        }
        result = validator.validate(data)
        assert result.is_valid
        assert not result.errors

    def test_invalid_dtc_format(self, validator):
        """测试无效的DTC格式"""
        data = {
            "dtc_codes": [{"code": "X0135", "status": "Active"}]
        }
        result = validator.validate(data)
        assert not result.is_valid
        assert any("无效的 DTC 格式" in err for err in result.errors)

    def test_invalid_dtc_status(self, validator):
        """测试无效的DTC状态"""
        data = {
            "dtc_codes": [{"code": "P0135", "status": "Unknown"}]
        }
        result = validator.validate(data)
        assert not result.is_valid
        assert any("无效的 DTC 格式" in err for err in result.errors)

    def test_incomplete_ecu_state(self, validator):
        """测试不完整的ECU状态"""
        data = {
            "ecu_states": [{"ecu_id": "ECU_ABC_123"}]  # 缺少 status
        }
        result = validator.validate(data)
        assert result.is_valid  # 缺少状态是warning
        assert not result.errors
        assert any("ECU 状态信息不完整" in warn for warn in result.warnings)

    def test_valid_mixed_data(self, validator):
        """测试有效的混合数据"""
        data = {
            "dtc_codes": [{"code": "P0135", "status": "Stored"}],
            "ecu_states": [{"ecu_id": "ECU_ABC_123", "status": "Online"}],
            "freeze_frames": [{"dtc_code": "P0135", "rpm": 3000}]
        }
        result = validator.validate(data)
        assert result.is_valid
        assert not result.errors
        assert not result.warnings


class TestStateTransitionValidator:
    @pytest.fixture
    def validator(self):
        return StateTransitionValidator()

    def test_valid_transition(self, validator):
        """测试有效的状态转换"""
        result = validator.validate_transition("input_validator", "parser", {})
        assert result.is_valid
        assert not result.errors

    def test_invalid_source_node(self, validator):
        """测试未知的源节点"""
        result = validator.validate_transition("unknown_node", "parser", {})
        assert not result.is_valid
        assert any("未知源节点" in err for err in result.errors)

    def test_invalid_target_node(self, validator):
        """测试不允许的转换目标节点"""
        result = validator.validate_transition("input_validator", "rca_engine", {})
        assert not result.is_valid
        assert any("不允许的转换" in err for err in result.errors)

    def test_report_generator_end(self, validator):
        """测试报告生成器没有后续节点"""
        result = validator.validate_transition("report_generator", "any", {})
        assert not result.is_valid
        assert any("不允许的转换" in err for err in result.errors)
