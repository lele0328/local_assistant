"""MCP 工具服务（stdio）—— mcp 2.x 版本

把项目里已有的 5 个工具以「薄封装」暴露为标准 MCP server：
工具的业务逻辑一行不改，本文件只负责协议层（工具签名、schema、参数约束）。

注意：mcp 2.x 把 FastMCP 改名为 MCPServer（from mcp.server.mcpserver import MCPServer）。
沿用 v1 的 from mcp.server.fastmcp import FastMCP 会直接 ModuleNotFoundError。

启动（stdio）：
    python mcp_server.py

自检（真实 stdio 拉起来、列工具、逐个调用、验证越界防护）：
    python mcp_client.py
"""
import sys
from pathlib import Path
from typing import Literal

from mcp.server.mcpserver import MCPServer

# file_search 的 directory 参数来自外部客户端，属于不可信输入。
# 必须限制在允许范围内，否则一个 "../../.." 就能把整台机器的文件列出来。
BASE_DIR = Path(__file__).resolve().parent

from tools.calculator import CalculatorTool
from tools.file_search import FileSearchTool
from tools.notes import NotesTool
from tools.weather import WeatherTool
from tools.web_search import WebSearchTool

calculator_tool = CalculatorTool()
file_search_tool = FileSearchTool()
notes_tool = NotesTool()
weather_tool = WeatherTool()
web_search_tool = WebSearchTool()

mcp = MCPServer("local-assistant-tools")


@mcp.tool()
def calculator(expression: str) -> str:
    """数学计算：支持 + - * / // % ** 与括号，只接受纯数学表达式。

    Args:
        expression: 数学表达式，如 "3+5*2"、"(10+5)/3"、"2**10"
    """
    return calculator_tool.execute(expression=expression)


@mcp.tool()
def file_search(keyword: str, directory: str = ".") -> str:
    """按文件名关键词搜索本地文件，搜索范围限定在项目目录内。

    Args:
        keyword: 文件名关键词（子串匹配）
        directory: 相对项目根目录的子目录，默认当前目录；不允许指向项目外
    """
    target = (BASE_DIR / directory).resolve()
    # 绝对路径经 / 运算后会整体替换，is_relative_to 会把它挡下来
    if not target.is_relative_to(BASE_DIR):
        return "拒绝：只能搜索项目目录内部"
    return file_search_tool.execute(keyword=keyword, directory=str(target))


@mcp.tool()
def notes(action: Literal["create", "list", "read", "delete"], title: str = "", content: str = "") -> str:
    """笔记管理：创建 / 列出 / 读取 / 删除本地笔记。

    Args:
        action: create 需要 title + content；read / delete 需要 title；list 不需要参数
        title: 笔记标题
        content: 笔记内容（仅 create 使用）
    """
    return notes_tool.execute(action=action, title=title, content=content)


@mcp.tool()
def weather(city: str) -> str:
    """查询城市天气（依赖高德地图 API Key）。

    Args:
        city: 城市名，如 "北京"、"上海"
    """
    return weather_tool.execute(city=city)


@mcp.tool()
def web_search(query: str) -> str:
    """从互联网检索最新信息（Bing RSS，无需额外 Key）。

    Args:
        query: 搜索关键词
    """
    return web_search_tool.execute(query=query)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    sys.exit(main())
