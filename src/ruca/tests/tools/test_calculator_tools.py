"""
Tests for calculator_tools (CalculatorTool)
Testing safe expression evaluation with pytest and mock
"""

import math
from unittest.mock import patch

from ruca.tools.caclculator_tools import CalculatorTool


class TestCalculatorTool:
    """Test suite for CalculatorTool"""

    def test_get_tools_metadata(self):
        """Test that get_tools_metadata returns correct structure"""
        metadata = CalculatorTool.get_tools_metadata()

        assert isinstance(metadata, list)
        assert len(metadata) == 1

        calculator_tool = metadata[0]
        assert calculator_tool["name"] == "calculator"
        assert "description" in calculator_tool
        assert "parameters" in calculator_tool

    def test_safe_eval_simple_addition(self):
        """Test simple addition calculation"""
        result = CalculatorTool.calculate("2 + 3")

        assert result["success"] is True
        assert result["result"] == 5

    def test_safe_eval_simple_subtraction(self):
        """Test simple subtraction calculation"""
        result = CalculatorTool.calculate("10 - 3")

        assert result["success"] is True
        assert result["result"] == 7

    def test_safe_eval_multiplication(self):
        """Test multiplication calculation"""
        result = CalculatorTool.calculate("4 * 5")

        assert result["success"] is True
        assert result["result"] == 20

    def test_safe_eval_division(self):
        """Test division calculation"""
        result = CalculatorTool.calculate("15 / 3")

        assert result["success"] is True
        assert result["result"] == 5.0

    def test_safe_eval_floor_division(self):
        """Test floor division calculation"""
        result = CalculatorTool.calculate("17 // 5")

        assert result["success"] is True
        assert result["result"] == 3

    def test_safe_eval_modulo(self):
        """Test modulo operation"""
        result = CalculatorTool.calculate("17 % 5")

        assert result["success"] is True
        assert result["result"] == 2

    def test_safe_eval_power(self):
        """Test power operation"""
        result = CalculatorTool.calculate("2 ** 3")

        assert result["success"] is True
        assert result["result"] == 8

    def test_safe_eval_complex_expression(self):
        """Test complex mathematical expression"""
        result = CalculatorTool.calculate("(2 + 3) * 4 - 5")

        assert result["success"] is True
        assert result["result"] == 15

    def test_safe_eval_negative_numbers(self):
        """Test calculation with negative numbers"""
        result = CalculatorTool.calculate("-5 + 3")

        assert result["success"] is True
        assert result["result"] == -2

    def test_safe_eval_unary_plus(self):
        """Test unary plus operator"""
        result = CalculatorTool.calculate("+5")

        assert result["success"] is True
        assert result["result"] == 5

    def test_safe_eval_unary_minus(self):
        """Test unary minus operator"""
        result = CalculatorTool.calculate("-(10 + 5)")

        assert result["success"] is True
        assert result["result"] == -15

    def test_safe_eval_math_functions(self):
        """Test math module functions"""
        # Test sqrt
        result = CalculatorTool.calculate("sqrt(16)")
        assert result["success"] is True
        assert result["result"] == 4.0

        # Test sin
        result = CalculatorTool.calculate("sin(0)")
        assert result["success"] is True
        assert result["result"] == 0.0

    def test_safe_eval_math_constants(self):
        """Test math constants"""
        # Test pi
        result = CalculatorTool.calculate("pi")
        assert result["success"] is True
        assert abs(result["result"] - math.pi) < 0.0001

        # Test e
        result = CalculatorTool.calculate("e")
        assert result["success"] is True
        assert abs(result["result"] - math.e) < 0.0001

    def test_safe_eval_abs_function(self):
        """Test abs function"""
        result = CalculatorTool.calculate("abs(-42)")

        assert result["success"] is True
        assert result["result"] == 42

    def test_safe_eval_round_function(self):
        """Test round function"""
        result = CalculatorTool.calculate("round(3.7)")

        assert result["success"] is True
        assert result["result"] == 4

    def test_safe_eval_disallowed_function(self):
        """Test that disallowed functions are rejected"""
        result = CalculatorTool.calculate("__import__('os')")

        assert result["success"] is False
        assert "error" in result

    def test_safe_eval_injection_attempt(self):
        """Test that injection attempts are blocked"""
        result = CalculatorTool.calculate("exec('print(1)')")

        assert result["success"] is False
        assert "error" in result

    def test_safe_eval_division_by_zero(self):
        """Test handling of division by zero"""
        result = CalculatorTool.calculate("1 / 0")

        assert result["success"] is False
        assert "error" in result

    def test_safe_eval_invalid_expression(self):
        """Test handling of invalid expressions"""
        result = CalculatorTool.calculate("undefined_var + 5")

        assert result["success"] is False
        assert "error" in result or "message" in result

    def test_safe_eval_returns_dict(self):
        """Test that calculate returns a dictionary"""
        result = CalculatorTool.calculate("5 + 3")

        assert isinstance(result, dict)
        assert "success" in result

    def test_safe_eval_floating_point_operations(self):
        """Test floating point calculations"""
        result = CalculatorTool.calculate("0.1 + 0.2")

        assert result["success"] is True
        # Allow for floating point precision
        assert abs(result["result"] - 0.3) < 0.0001

    def test_safe_eval_nested_functions(self):
        """Test nested function calls"""
        result = CalculatorTool.calculate("sqrt(abs(-16))")

        assert result["success"] is True
        assert result["result"] == 4.0

    def test_safe_eval_large_numbers(self):
        """Test calculations with large numbers"""
        result = CalculatorTool.calculate("999999 * 999999")

        assert result["success"] is True
        assert result["result"] == 999999 * 999999

    def test_safe_eval_logarithm(self):
        """Test logarithm functions"""
        result = CalculatorTool.calculate("log10(100)")

        assert result["success"] is True
        assert result["result"] == 2.0

    def test_safe_eval_trigonometric(self):
        """Test trigonometric functions"""
        # cos(0) = 1
        result = CalculatorTool.calculate("cos(0)")
        assert result["success"] is True
        assert abs(result["result"] - 1.0) < 0.0001

    @patch("ruca.tools.caclculator_tools._eval_node")
    def test_safe_eval_with_mock(self, mock_eval):
        """Test calculate with mocked _eval_node"""
        mock_eval.return_value = 42

        CalculatorTool.calculate("mock_expression")

    def test_safe_eval_zero_result(self):
        """Test expression that results in zero"""
        result = CalculatorTool.calculate("5 - 5")

        assert result["success"] is True
        assert result["result"] == 0

    def test_safe_eval_returns_success_key(self):
        """Test that result always contains success key"""
        test_expressions = ["2 + 2", "1 / 0", "invalid", "sin(0)"]

        for expr in test_expressions:
            result = CalculatorTool.calculate(expr)
            assert "success" in result
            assert isinstance(result["success"], bool)
