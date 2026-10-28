from mcp.server.fastmcp import FastMCP
from pathlib import Path
from typing import Literal
from tools.notes import NotesTool
from tools.file_search import FileSearchTool
from tools.calculator import CalculatorTool
from tools.weather import WeatherTool
from tools.web_search import WebSearchTool
mcp=FastMCP("local-assistant-tools")
notes_tool = NotesTool()
file_search_tool = FileSearchTool()
calculator_tool = CalculatorTool()
weather_tool = WeatherTool()
web_search_tool = WebSearchTool()
BASE_DIR = Path(__file__).resolve().parent
@mcp.tool()
def calculator(expression: str) -> str:
    """数学计算工具，支持加减乘除、幂运算、取余，只接受纯数学表达式"""
    return calculator_tool.execute(expression=expression)
@mcp.tool()
def file_search(keyword: str, directory: str = ".")->str:
    """按文件名关键词搜索本地文件"""
    target = (BASE_DIR / directory).resolve()
    if not target.is_relative_to(BASE_DIR):
        return "拒绝：只能搜索项目目录内部"
    return file_search_tool.execute(keyword=keyword,directory=str(target))
@mcp.tool()
def notes(action:Literal["create", "list", "read", "delete"], title: str = "", content: str = "")->str:
    """笔记管理工具：action支持create / list / read / delete"""
    return notes_tool.execute(action=action,title=title,content=content)
@mcp.tool()
def weather(city: str)->str:
    """查询任意城市的天气，输入城市名"""
    return weather_tool.execute(city=city)
@mcp.tool()
def web_search(query: str)->str:
    """从互联网搜索最新信息，输入关键词"""
    return web_search_tool.execute(query=query)

if __name__=="__main__" :
    mcp.run()