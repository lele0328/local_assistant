"""
工具基类和工具注册器
所有工具继承Tool基类，通过ToolRegistry统一管理
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class Tool(ABC):
    """工具基类，所有工具必须继承"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具，子类必须实现"""
        pass

    def get_schema(self) -> dict:
        """返回工具的参数schema，子类可以覆盖"""
        return {"type": "object", "properties": {}, "required": []}


class ToolRegistry:
    """工具注册器"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool

    def execute(self, name: str, args: dict) -> str:
        """执行指定工具"""
        if name not in self.tools:
            return "未知工具"
        return self.tools[name].execute(**args)

    def get_all_schemas(self) -> List[dict]:
        """获取所有工具的schema列表"""
        schemas = []
        for tool in self.tools.values():
            schemas.append(tool.get_schema())
        return schemas

    def get_tool_names(self) -> List[str]:
        """获取所有工具名称"""
        return list(self.tools.keys())
