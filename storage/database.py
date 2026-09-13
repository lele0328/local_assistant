"""
本地数据库存储模块
管理对话历史和用户数据的本地存储
"""
import json
import os
from typing import List, Dict, Optional
from datetime import datetime


class Database:
    """本地JSON数据库"""

    def __init__(self, db_path: str = "./data/database.json"):
        self.db_path = db_path
        self._ensure_file()

    def _ensure_file(self):
        """确保数据库文件存在"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump({"conversations": [], "notes": []}, f, ensure_ascii=False, indent=2)

    def _load(self) -> dict:
        """加载数据库"""
        with open(self.db_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict):
        """保存数据库"""
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_conversation(self, user_msg: str, assistant_msg: str):
        """保存对话记录"""
        data = self._load()
        data["conversations"].append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_msg,
            "assistant": assistant_msg
        })
        self._save(data)

    def get_conversations(self, limit: int = 10) -> List[Dict]:
        """获取最近的对话记录"""
        data = self._load()
        return data["conversations"][-limit:]

    def clear_conversations(self):
        """清空对话记录"""
        data = self._load()
        data["conversations"] = []
        self._save(data)
