"""
Agent核心模块
ReAct循环 + Function Calling
"""
import json
from typing import List, Dict, Optional
from openai import OpenAI

from config import config


class Agent:
    """ReAct Agent - 思考→行动→观察 循环"""

    def __init__(self, system_prompt: str = None):
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.model = config.model_id
        self.system_prompt = system_prompt or config.system_prompt
        self.tools: Dict = {}
        self.tool_schemas: List = []
        self.max_steps = config.max_steps

    def register_tool(self, name: str, func, description: str, params: dict):
        """注册工具"""
        self.tools[name] = {"func": func, "desc": description}
        self.tool_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": params
            }
        })

    def think(self, messages: List[Dict]):
        """调用LLM，返回response"""
        print(f"[Agent] 可用工具: {[t['function']['name'] for t in self.tool_schemas]}")
        print(f"[Agent] 发送消息数: {len(messages)}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                tools=self.tool_schemas,
                messages=messages,
                timeout=30,
                temperature=0.2
            )
            msg = response.choices[0].message
            print(f"[Agent] LLM返回 tool_calls: {msg.tool_calls}")
            print(f"[Agent] LLM返回 content前100字: {(msg.content or '')[:100]}")
            return response
        except Exception as e:
            print(f"[Agent] LLM调用失败: {e}")
            raise

    def act(self, tool_name: str, args: dict) -> str:
        """执行工具"""
        func = self.tools[tool_name]["func"]
        return func(**args)

    def run(self, question: str) -> str:
        """ReAct主循环 - 带搜索意图预判"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        # 预判：如果用户想搜索，直接先调 web_search
        search_keywords = ["搜索", "查找", "查一下", "帮我找", "最新", "是什么",
                           "新闻", "攻略", "推荐", "怎么回事", "介绍", "了解"]
        want_search = any(kw in question for kw in search_keywords)

        if want_search and "web_search" in self.tools:
            print("[Agent] 检测到搜索意图，直接调用 web_search")
            result = self.act("web_search", {"query": question})
            print(f"[Agent] web_search 结果: {result[:200]}...")
            messages.append({
                "role": "system",
                "content": f"以下是网络搜索结果，请基于这些信息回答用户：\n{result}"
            })

        step = 0
        while step < self.max_steps:
            step += 1
            print(f"\n--- 第{step}轮 ---")
            response = self.think(messages)
            msg = response.choices[0].message
            if not msg.tool_calls:
                return msg.content
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = self.act(name, args)
                print(f"调用工具: {name}({args})")
                print(f"执行结果: {result}")
                messages.append({
                    "role": "assistant",
                    "tool_calls": [tool_call],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        return "轮数超限，强制结束"
