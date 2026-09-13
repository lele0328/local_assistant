"""
计算器工具
"""
from tools.base import Tool


class CalculatorTool(Tool):
    """计算器工具，支持加减乘除"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="数学计算工具，支持加减乘除运算"
        )

    def execute(self, expression: str) -> str:
        """执行数学表达式"""
        try:
            result = eval(expression)
            return str(result)
        except:
            return "计算错误"

    def get_schema(self) -> dict:
        """返回参数schema"""
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式，如 3+5*2"}
            },
            "required": ["expression"]
        }
