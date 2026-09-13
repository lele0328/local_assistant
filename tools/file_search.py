"""
本地文件搜索工具
"""
import os
from tools.base import Tool


class FileSearchTool(Tool):
    """本地文件搜索工具"""

    def __init__(self):
        super().__init__(
            name="file_search",
            description="搜索本地文件，按文件名查找"
        )

    def execute(self, keyword: str, directory: str = ".") -> str:
        """搜索文件"""
        results = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if keyword in f:
                    results.append(os.path.join(root, f))
                    if len(results) >= 20:
                        break
            if len(results) >= 20:
                break
        if not results:
            return "未找到匹配文件"
        return "\n".join(results)

    def get_schema(self) -> dict:
        """返回参数schema"""
        return {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "搜索关键词"},
                "directory": {"type": "string", "description": "搜索目录，默认当前目录"}
            },
            "required": ["keyword"]
        }
