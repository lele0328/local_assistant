"""
上下文工程模块
GSSC流水线：Gather → Select → Structure → Compress
"""
from typing import List, Dict


class ContextBuilder:
    """上下文构建器"""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def gather(self, question: str, history: List[Dict], docs: List = None) -> Dict:
        """收集所有可能的上下文来源"""
        return {"question": question, "history": history, "docs": docs or []}

    def select(self, context: Dict, top_k: int = 3) -> Dict:
        """筛选最相关的上下文"""
        n = 5
        docs = context["docs"][:top_k]
        history = context["history"][-n:]
        return {"history": history, "docs": docs}

    def structure(self, context: Dict) -> str:
        """把上下文结构化为prompt文本"""
        prompt = []
        docs = context["docs"]
        historys = context["history"]
        for doc in docs:
            prompt.append(doc["content"])
        for history in historys:
            prompt.append(history["content"])
        return "\n".join(prompt)

    def compress(self, text: str) -> str:
        """压缩过长的上下文"""
        if len(text) > self.max_tokens * 4:
            return text[:self.max_tokens * 4] + "...(已截断)"
        else:
            return text

    def build(self, question: str, history: List[Dict], docs: List = None) -> str:
        """完整GSSC流水线"""
        content = self.gather(question, history, docs)
        selected = self.select(content)
        text = self.structure(selected)
        result = self.compress(text)
        return result
