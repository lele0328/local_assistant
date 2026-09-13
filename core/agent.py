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
        response = self.client.chat.completions.create(
            model=self.model,
            tools=self.tool_schemas,
            messages=messages,
            timeout=30,
            temperature=0.2
        )
        return response

    def act(self, tool_name: str, args: dict) -> str:
        """执行工具"""
        func = self.tools[tool_name]["func"]
        return func(**args)

    def run(self, question: str) -> str:
        """ReAct主循环"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

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
