"""
Рабочий инструмент: calculator
Безопасно вычисляет математические выражения, используя ast-парсинг и белый список операций/функций.
"""

from typing import Dict, Any, List
import ast
import operator as op
import math

# Допустимые бинарные операции
_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

# Допустимые унарные операции
_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda x: x,
    ast.USub: lambda x: -x,
}

# Разрешённые функции и константы (включая те, что в math)
_ALLOWED_NAMES = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
# Добавим пару удобных алиасов
_ALLOWED_NAMES.update({
    "abs": abs,
    "round": round,
    "pi": math.pi,
    "e": math.e
})


def _eval_node(node):
    """Рекурсивно вычисляет AST-узел в безопасном окружении."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):  # Python 3.8+: числа и константы
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported constant type")

    if isinstance(node, ast.Num):  # старый узел для чисел
        return node.n

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
        # вызов функции: имя должен быть простым идентификатором
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in _ALLOWED_NAMES:
                raise ValueError(f"Function '{func_name}' is not allowed")
            func = _ALLOWED_NAMES[func_name]
            args = [_eval_node(a) for a in node.args]
            # не поддерживаем kwargs, *args, comprehension и т.д.
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
    @staticmethod
    def get_tools_metadata() -> List[Dict[str, Any]]:
        return [
            {
                "name": "calculator",
                "description": "Вычисляет математическое выражение безопасно (поддерживаются + - * / // % **, скобки, функции из math).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Математическое выражение, например '2*(3+sin(pi/2))'"},
                        "precision": {"type": "integer", "description": "Опционально: число знаков после запятой в результате"}
                    },
                    "required": ["expression"]
                }
            }
        ]

    @staticmethod
    def calculate(expression: str, precision: int = None) -> Dict[str, Any]:
        """Безопасно вычисляет expression и возвращает результат или ошибку."""
        if not isinstance(expression, str) or not expression.strip():
            return {"success": False, "error": "invalid_input", "message": "Empty or non-string expression"}

        try:
            # Парсим выражение в AST (режим 'eval' — только выражения)
            parsed = ast.parse(expression, mode="eval")
            # Пройдём по дереву и вычислим
            result = _eval_node(parsed)
            if isinstance(result, float) and precision is not None:
                try:
                    precision = int(precision)
                    result = round(result, precision)
                except Exception:
                    # если precision неверный — игнорируем округление
                    pass

            return {"success": True, "expression": expression, "result": result}
        except ZeroDivisionError:
            return {"success": False, "error": "zero_division", "message": "Division by zero"}
        except Exception as e:
            return {"success": False, "error": "eval_error", "message": str(e)}

def register_calculator(tool_registry):
    """
    Регистрирует инструмент calculator в tool_registry.
    """
    metas = CalculatorTool.get_tools_metadata()
    # единственный исполнитель
    executors = {"calculator": CalculatorTool.calculate}
    for meta in metas:
        name = meta["name"]
        tool_registry.register_tool(name, meta, executors[name])
    print("✅ Зарегистрирован инструмент calculator")
