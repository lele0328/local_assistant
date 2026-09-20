"""
网络搜索工具
使用Bing RSS搜索，国内可正常访问，无需API Key
"""
import requests
import xml.etree.ElementTree as ET
from tools.base import Tool


class WebSearchTool(Tool):
    """网络搜索工具，从Bing获取搜索结果"""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="从互联网搜索最新信息，输入关键词即可获取搜索结果"
        )

    def execute(self, query: str) -> str:
        """执行搜索：用Bing RSS格式"""
        try:
            resp = requests.get(
                "https://cn.bing.com/search",
                params={"q": query, "format": "rss", "count": "10"},
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept-Language": "zh-CN,zh;q=0.9"
                }
            )
            resp.raise_for_status()

            results = self._parse_rss(resp.text)

            if results:
                lines = [f"搜索'{query}'的结果："]
                for i, item in enumerate(results[:5], 1):
                    lines.append(f"{i}. {item['title']}")
                    if item.get("description"):
                        lines.append(f"   {item['description'][:200]}")
                return "\n".join(lines)

            return f"未找到关于'{query}'的相关信息"

        except Exception as e:
            return f"搜索出错: {e}"

    def _parse_rss(self, xml_text: str) -> list:
        """解析Bing RSS XML"""
        results = []
        root = ET.fromstring(xml_text)
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc = (item.findtext("description") or "").strip()
            if title:
                results.append({"title": title, "url": link, "description": desc})
        return results

    def get_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
