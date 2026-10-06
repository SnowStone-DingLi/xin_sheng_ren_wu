"""计算器工具：AST 白名单求值，绝不直接 eval（M1）。"""
import ast
import math

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "安全的数学计算器。支持 + - * / // % ** 、括号、一元负号，"
            "以及 math 模块函数（sqrt, log, log10, sin, cos, tan, exp, "
            "pow, floor, ceil, gcd, factorial 等）和常量 pi、e。"
            "参数 expression 是一个合法 Python 算术表达式字符串。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "例如 '2 + 3 * 4' 或 'sqrt(2026)'",
                }
            },
            "required": ["expression"],
        },
    },
}

_ALLOWED_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
    "log10": math.log10, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "asin": math.asin, "acos": math.acos,
    "atan": math.atan, "exp": math.exp, "pow": math.pow,
    "floor": math.floor, "ceil": math.ceil, "gcd": math.gcd,
    "factorial": math.factorial, "fabs": math.fabs, "radians": math.radians,
    "degrees": math.degrees, "abs": abs, "round": round, "min": min,
    "max": max, "sum": sum,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Load, ast.UnaryOp,
)


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不允许的常量: {node.value!r}")
    if isinstance(node, ast.BinOp):
        ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
               ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
               ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
               ast.Pow: lambda a, b: a ** b}
        if type(node.op) not in ops:
            raise ValueError("不允许的运算符")
        left, right = _eval(node.left), _eval(node.right)
        # 禁止超大指数，避免卡死
        if isinstance(node.op, ast.Pow) and (abs(right) > 1000):
            raise ValueError("指数过大")
        return ops[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +_eval(node.operand)
        raise ValueError("不允许的一元运算")
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return _ALLOWED_CONSTS[node.id]
        raise ValueError(f"未知变量: {node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("只允许普通函数调用")
        fn = _ALLOWED_FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"不允许的函数: {node.func.id}")
        return fn(*[_eval(a) for a in node.args],
                  **{kw.arg: _eval(kw.value) for kw in node.keywords})
    raise ValueError(f"不允许的语法: {type(node).__name__}")


def format_result(x):
    if isinstance(x, float):
        if x.is_integer() and abs(x) < 1e16:
            return str(int(x))
        # 统一保留 6 位小数，便于 agent 直接引用精度
        return f"{x:.6f}"
    return str(x)


def run(args: dict) -> str:
    expr = args["expression"]
    tree = ast.parse(str(expr), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"表达式包含不允许的语法: {type(node).__name__}")
    value = _eval(tree)
    return f"{expr} = {format_result(value)}"
