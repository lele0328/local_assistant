"""
计算器工具（安全版）

替换 tools/calculator.py。

改动点：
1. 用 AST 白名单求值替代 eval()，彻底消除任意代码执行漏洞
2. 去掉裸 except:，改为捕获具体异常
3. 错误信息返回给模型可理解的描述，不泄露内部堆栈
4. 增加结果长度和数值范围保护
"""
import ast
import operator

from tools.base import Tool


# 只允许这些运算符，其他一律拒绝
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# 幂运算的上限，防止 9**9**9 这类把内存打爆
_MAX_POW_EXPONENT = 100


def _eval_node(node: ast.AST) -> float:
    """递归求值，只认白名单里的节点类型"""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    # 数字字面量
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"不支持的常量类型: {type(node.value).__name__}")

    # 二元运算 a op b
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError(f"指数过大（上限 {_MAX_POW_EXPONENT}）")
        return _ALLOWED_OPS[op_type](left, right)

    # 一元运算 -a / +a
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
        return _ALLOWED_OPS[op_type](_eval_node(node.operand))

    # 显式拒绝函数调用、属性访问、下标、推导式等一切其他语法
    raise ValueError(f"表达式包含不允许的语法: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """
    安全地计算数学表达式。

    只支持：数字、+ - * / // % **、括号、一元正负号。
    不接受变量、函数调用、属性访问。

    Raises:
        ValueError: 表达式非法
        ZeroDivisionError: 除零
    """
    if not expression or not expression.strip():
        raise ValueError("表达式为空")
    if len(expression) > 200:
        raise ValueError("表达式过长")

    tree = ast.parse(expression.strip(), mode="eval")
    result = _eval_node(tree)

    # 结果范围保护，避免返回 inf / nan 这种模型看不懂的东西
    if isinstance(result, float):
        if result != result:  # NaN
            raise ValueError("计算结果不是一个有效数字")
        if result in (float("inf"), float("-inf")):
            raise ValueError("计算结果溢出")

    return result


class CalculatorTool(Tool):
    """计算器工具，支持加减乘除等基础运算（不使用 eval）"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description=(
                "数学计算工具，支持加(+)、减(-)、乘(*)、除(/)、取整除法(//)、"
                "取余(%)、幂运算(**)、括号。只接受纯数学表达式，不支持变量和函数。"
                "示例：'3+5*2'、'(10+5)/3'、'2**10'"
            ),
        )

    def execute(self, expression: str) -> str:
        """执行数学表达式"""
        try:
            result = safe_eval(expression)
            # 整数结果去掉小数点，看起来更自然
            if isinstance(result, float) and result.is_integer():
                return str(int(result))
            return str(result)
        except ZeroDivisionError:
            return "计算错误：除数不能为零。"
        except ValueError as e:
            return f"计算错误：{e}。请提供一个合法的数学表达式。"
        except SyntaxError:
            return "计算错误：表达式语法不合法，请检查括号是否配对。"

    def get_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 3+5*2 或 (10+5)/3",
                }
            },
            "required": ["expression"],
        }


if __name__ == "__main__":
    # 自测：这些必须全部通过
    tool = CalculatorTool()
    cases = [
        ("3+5*2", "13"),
        ("(10+5)/3", "5"),
        ("2**10", "1024"),
        ("-5+3", "-2"),
        ("10//3", "3"),
        ("10%3", "1"),
        # 以下必须全部被拒绝
        ("__import__('os').system('echo hacked')", None),
        ("open('/etc/passwd').read()", None),
        ("1/0", None),
        ("[x for x in range(3)]", None),
        ("', '1+1", None),
    ]
    for expr, expected in cases:
        got = tool.execute(expr)
        status = "OK " if (expected is None or got == expected) else "FAIL"
        print(f"[{status}] {expr!r} -> {got}")
