"""
本地数据库存储模块
管理对话历史和用户数据的本地存储
"""
import json
import os
from typing import List, Dict, Optional
from datetime import datetime


class DatabaseError(Exception):
    """数据库异常"""

    def __init__(self, message):
        super().__init__(message)


class Database:
    """本地JSON数据库"""

    def __init__(self, db_path: str = "./data/database.json"):
        self._db_path = db_path
        self._ensure_file()

    @property
    def db_path(self):
        return self._db_path

    @classmethod
    def from_config(cls, config):
        return cls(config.DB_PATH)

    @staticmethod
    def is_valid_path(path):
        return path.endswith(".json")

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        if not os.path.exists(self._db_path):
            with open(self._db_path, "w", encoding="utf-8") as f:
                json.dump({"conversations": [], "notes": []}, f, ensure_ascii=False, indent=2)

    def _load(self) -> dict:
        try:
            with open(self._db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"conversations": [], "notes": []}

    def _save(self, data: dict):
        try:
            with open(self._db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except PermissionError:
            raise DatabaseError("写入失败: 权限不足")

    def save_conversation(self, user_msg: str, assistant_msg: str):
        data = self._load()
        data["conversations"].append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_msg,
            "assistant": assistant_msg
        })
        self._save(data)

    def get_conversations(self, limit: int = 10) -> List[Dict]:
        data = self._load()
        return data["conversations"][-limit:]

    def clear_conversations(self):
        data = self._load()
        data["conversations"] = []
        self._save(data)

    def __str__(self):
        data = self._load()
        return f"Database(path={self._db_path}, records={len(data['conversations'])})"
