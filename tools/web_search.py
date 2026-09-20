"""
网络搜索工具
使用DuckDuckGo搜索API，免费无需API Key
"""
import requests
from tools.base import Tool


class WebSearchTool(Tool):
    """网络搜索工具，从DuckDuckGo获取搜索结果"""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="从互联网搜索最新信息，输入关键词即可获取搜索结果"
        )

    def execute(self, query: str) -> str:
        """执行搜索"""
        try:
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                },
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            data = resp.json()

            results = []

            # AbstractText是最相关的摘要
            if data.get("AbstractText"):
                results.append(f"摘要: {data['AbstractText']}")
                if data.get("AbstractURL"):
                    results.append(f"来源: {data['AbstractURL']}")

            # Answer是直接回答
            if data.get("Answer"):
                results.append(f"回答: {data['Answer']}")

            # Definition是定义
            if data.get("Definition"):
                results.append(f"定义: {data['Definition']}")

            # RelatedTopics是相关结果
            related = data.get("RelatedTopics", [])
            if related:
                results.append("相关结果:")
                for i, item in enumerate(related[:5], 1):
                    if isinstance(item, dict) and item.get("Text"):
                        results.append(f"  {i}. {item['Text'][:200]}")

            if results:
                return "\n".join(results)
            return f"未找到关于'{query}'的相关信息"

        except Exception as e:
            return f"搜索出错: {e}"

    def get_schema(self) -> dict:
        """返回参数schema"""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
