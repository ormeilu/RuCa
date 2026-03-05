"""Calculator tool for the LLM agent benchmark.

Safely evaluates mathematical expressions via AST parsing and a whitelist
of allowed operations and functions.
"""

import ast
import math
import operator as op
from typing import Any

# Allowed binary operations
_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

# Allowed unary operations
_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda x: x,
    ast.USub: lambda x: -x,
}

# Allowed functions and constants (including those in math)
_ALLOWED_NAMES = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
# Add a few convenience aliases
_ALLOWED_NAMES.update({"abs": abs, "round": round, "pi": math.pi, "e": math.e})


def _eval_node(node) -> Any:
    """Recursively evaluate an AST node in a safe sandbox."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):  # Python 3.8+: numbers and constants
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported constant type")

    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BINOPS:
            raise ValueError(f"Operator {type(node.op).__name__} not allowed")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_BINOPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARYOPS:
            raise ValueError(f"Unary operator {type(node.op).__name__} not allowed")
        operand = _eval_node(node.operand)
        return _ALLOWED_UNARYOPS[type(node.op)](operand)

    if isinstance(node, ast.Call):
        # Function call: the name must be a plain identifier
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in _ALLOWED_NAMES:
                raise ValueError(f"Function '{func_name}' is not allowed")
            func = _ALLOWED_NAMES[func_name]
            args = [_eval_node(a) for a in node.args]
            # kwargs, *args, comprehensions, etc. are not supported
            return func(*args)
        else:
            raise ValueError("Only direct function names are allowed in calls")

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Name '{node.id}' is not allowed")

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt) for elt in node.elts)

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


class CalculatorTool:
    """Safe mathematical-expression evaluator."""

    @staticmethod
    def get_tools_metadata() -> list[dict[str, Any]]:
        """Return OpenAI-compatible metadata for the calculator tool."""
        return [
            {
                "name": "calculator",
                "description": "Вычисляет математическое выражение безопасно (поддерживаются + - * / // % **, скобки, функции из math).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Математическое выражение, например '2*(3+sin(pi/2))'",
                        },
                        "precision": {
                            "type": "integer",
                            "description": "Опционально: число знаков после запятой в результате",
                        },
                    },
                    "required": ["expression"],
                },
            }
        ]

    @staticmethod
    def calculate(expression: str, precision: int = None) -> dict[str, Any]:
        """Safely evaluate *expression* and return the result or an error."""
        if not isinstance(expression, str) or not expression.strip():
            return {"success": False, "error": "invalid_input", "message": "Empty or non-string expression"}

        try:
            # Parse the expression into an AST ('eval' mode — expressions only)
            parsed = ast.parse(expression, mode="eval")
            # Walk the tree and compute the result
            result = _eval_node(parsed)
            if isinstance(result, float) and precision is not None:
                try:
                    precision = int(precision)
                    result = round(result, precision)
                except Exception:
                    # If precision is invalid — skip rounding
                    pass

            return {"success": True, "expression": expression, "result": result}
        except ZeroDivisionError:
            return {"success": False, "error": "zero_division", "message": "Division by zero"}
        except Exception as e:
            return {"success": False, "error": "eval_error", "message": str(e)}


def register_calculator(tool_registry) -> None:
    """Register the calculator tool in the given tool registry.

    Args:
        tool_registry: Registry instance exposing a ``register_tool`` method.
    """
    metas = CalculatorTool.get_tools_metadata()
    executors: dict[str, Any] = {"calculator": CalculatorTool.calculate}
    for meta in metas:
        name = meta["name"]
        tool_registry.register_tool(name, meta, executors[name])
    print("✅ Зарегистрирован инструмент calculator")
