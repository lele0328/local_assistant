"""
笔记管理工具
支持增删改查本地笔记
"""
import json
import os
from typing import Optional
from tools.base import Tool
from config import config


class NotesTool(Tool):
    """笔记管理工具"""

    def __init__(self):
        super().__init__(
            name="notes",
            description="笔记管理工具，可以创建、查看、删除笔记"
        )
        self.notes_path = config.notes_path
        self._ensure_file()

    def _ensure_file(self):
        """确保笔记文件存在"""
        os.makedirs(os.path.dirname(self.notes_path), exist_ok=True)
        if not os.path.exists(self.notes_path):
            with open(self.notes_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load_notes(self) -> list:
        """加载所有笔记"""
        with open(self.notes_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_notes(self, notes: list):
        """保存笔记"""
        with open(self.notes_path, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)

    def execute(self, action: str, title: str = "", content: str = "") -> str:
        """执行笔记操作"""
        notes = self._load_notes()
        if action == "create":
            notes.append({"id": len(notes) + 1, "title": title, "content": content})
            self._save_notes(notes)
            return f"笔记已创建: {title}"
        elif action == "list":
            if not notes:
                return "暂无笔记"
            return "\n".join(f"- {n['title']}" for n in notes)
        elif action == "read":
            for n in notes:
                if n["title"] == title:
                    return n["content"]
            return "笔记不存在"
        elif action == "delete":
            for n in notes:
                if n["title"] == title:
                    notes.remove(n)
                    self._save_notes(notes)
                    return f"已删除: {title}"
            return "笔记不存在"
        return "未知操作"

    def get_schema(self) -> dict:
        """返回参数schema"""
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "操作：create/list/read/delete"},
                "title": {"type": "string", "description": "笔记标题"},
                "content": {"type": "string", "description": "笔记内容（create时需要）"}
            },
            "required": ["action"]
        }
