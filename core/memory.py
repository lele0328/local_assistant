"""
对话记忆管理模块
管理对话历史，支持滑动窗口和摘要压缩
"""
from typing import List, Dict, Optional


class MemoryManager:
    """对话记忆管理器"""

    def __init__(self, max_history: int = 20):
        self.history: List[Dict] = []
        self.max_history = max_history  # 最多保留多少条消息

    def add_user_message(self, content: str):
        """添加用户消息"""
        self.history.append({"role": "user", "content": content})
        return self.history

    def add_assistant_message(self, content: str):
        """添加助手消息"""
        self.history.append({"role": "assistant", "content": content})
        return self.history

    def add_tool_message(self, tool_call_id: str, content: str):
        """添加工具消息"""
        self.history.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})
        return self.history

    def get_messages(self) -> List[Dict]:
        """获取当前记忆（带滑动窗口截断）"""
        if len(self.history) > self.max_history:
            return self.history[-self.max_history:]
        else:
            return self.history

    def clear(self):
        """清空记忆"""
        self.history.clear()

    def get_summary(self) -> str:
        """获取记忆摘要（简化版：返回最后几条消息的内容）"""
        message = self.history[-3:]
        return "\n\n".join(msg["content"] for msg in message)
