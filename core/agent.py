"""
Agent 核心模块
ReAct 循环 + Function Calling + 记忆 + 轨迹记录

本次改动（相对初版）：
1. 接入 MemoryManager —— 原版每次 run() 都重建 messages，多轮对话实际不生效
2. 接入 TraceRecorder —— 记录每轮 think / 每次工具调用，便于事后排查
3. 移除「关键词意图识别预调工具」的绕过逻辑 —— 那会让 Agent 退化成 Workflow，
   改为通过「工具描述工程 + 系统提示」引导模型自主调用
4. 增加重复调用检测 —— 防止模型反复用相同参数调同一个工具直到耗尽轮数
5. 工具执行加超时与异常隔离 —— 单个工具失败不再中断整个循环
6. 工具结果按预算裁剪 —— 避免超长返回撑爆上下文
"""

import json
import time
from typing import Dict, List, Optional

from openai import OpenAI

from config import config
from core.memory import MemoryManager
from utils.logger import Logger
from utils.trace import TraceRecorder


# 单个工具结果注入上下文的最大字符数。
# 超长工具输出是「上下文被污染 / 效果下降」的常见原因，所以统一在这里裁剪。
MAX_TOOL_RESULT_CHARS = 3000

# 同一个「工具名 + 参数」连续重复调用多少次就判定为无效循环
MAX_REPEAT_CALLS = 2


class Agent:
    """ReAct Agent - 思考 → 行动 → 观察 循环"""

    def __init__(self, system_prompt: str = None, enable_memory: bool = True):
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.model = config.model_id
        self.system_prompt = system_prompt or config.system_prompt
        self.tools: Dict = {}
        self.tool_schemas: List = []
        self.max_steps = config.max_steps
        self.logger = Logger()

        self.enable_memory = enable_memory
        # 记忆：按「轮次」管理，保证 tool_calls 与 tool 响应成组
        self.memory = MemoryManager(
            max_turns=config.memory_max_turns,
            system_prompt=self.system_prompt,
        )

    # ---------- 工具注册 ----------

    def register_tool(self, name: str, func, description: str, params: dict):
        """注册工具"""
        self.tools[name] = {"func": func, "desc": description}
        self.tool_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": params,
            },
        })

    # ---------- ReAct 三个基本动作 ----------

    def think(self, messages: List[Dict]):
        """调用 LLM，返回 response"""
        return self.client.chat.completions.create(
            model=self.model,
            tools=self.tool_schemas,
            messages=messages,
            timeout=config.timeout,
            temperature=config.temperature,
        )

    def act(self, tool_name: str, args: dict) -> tuple:
        """
        执行工具，返回 (结果文本, 是否成功)。

        异常隔离：任何工具异常都被捕获并转成模型可理解的错误信息，
        绝不让单个工具失败中断整个 ReAct 循环。
        """
        entry = self.tools.get(tool_name)
        if entry is None:
            return f"错误：工具 '{tool_name}' 不存在。可用工具：{', '.join(self.tools)}", False

        try:
            result = entry["func"](**args)
            return self._clip_tool_result(result, tool_name), True
        except TypeError as e:
            # 参数不匹配 —— 把「哪里错了、应该用什么格式」告诉模型，让它自己纠正
            self.logger.error(f"tool {tool_name} 参数错误: {e!r}")
            schema = next(
                (s["function"]["parameters"] for s in self.tool_schemas
                 if s["function"]["name"] == tool_name),
                {},
            )
            return (
                f"错误：调用 {tool_name} 的参数不正确（{e}）。"
                f"正确的参数格式为：{json.dumps(schema, ensure_ascii=False)}。请修正后重试。",
                False,
            )
        except Exception as e:
            self.logger.error(f"tool {tool_name} 执行异常: {e!r}")
            # 给模型的信息要简短可行动，原始异常已写进日志
            return f"错误：{tool_name} 执行失败（{type(e).__name__}）。请换一种方式或告知用户暂时无法完成。", False

    @staticmethod
    def _clip_tool_result(result, tool_name: str) -> str:
        """裁剪过长的工具返回，避免撑爆上下文"""
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        if len(text) > MAX_TOOL_RESULT_CHARS:
            return (
                text[:MAX_TOOL_RESULT_CHARS]
                + f"\n...(结果过长已截断，原长度 {len(text)} 字符)"
            )
        return text

    # ---------- 主循环 ----------

    def run(self, question: str, use_memory: Optional[bool] = None) -> str:
        """
        执行一次完整的 ReAct 循环。

        Args:
            question: 用户输入
            use_memory: 是否使用多轮记忆，默认跟随 enable_memory
        """
        use_mem = self.enable_memory if use_memory is None else use_memory

        # 1) 构建消息：使用记忆（含 system prompt + 历史），而非每次重建
        if use_mem:
            messages = self.memory.build_messages(user_input=question)
            self.memory.add_user_message(question)
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ]

        # 2) 轨迹记录
        tracer = TraceRecorder(trace_dir=config.trace_path)
        tracer.start(question, mode="agent")

        # 3) 重复调用检测
        call_fingerprints: Dict[str, int] = {}

        step = 0
        try:
            while step < self.max_steps:
                step += 1
                print(f"\n--- 第{step}轮 ---")

                response = self.think(messages)
                tracer.log_think(step, messages, response)

                msg = response.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None)

                # 没有工具调用 → 这是最终答案
                if not tool_calls:
                    answer = msg.content or "(模型未返回内容)"
                    if use_mem:
                        self.memory.add_assistant_message(answer)
                    tracer.finish(answer, terminated_by="answer")
                    return answer

                # 先把带 tool_calls 的 assistant 消息入历史（必须）
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                })
                if use_mem:
                    # 只把 id 存进记忆，避免重复持有大对象
                    self.memory.add_assistant_tool_calls(
                        [type("TC", (), {"id": tc.id})() for tc in tool_calls]
                    )

                for tool_call in tool_calls:
                    name = tool_call.function.name

                    # --- 参数解析（模型可能返回非法 JSON）---
                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError as e:
                        err = (
                            f"错误：工具参数不是合法 JSON（{e}）。"
                            f"你返回的是：{tool_call.function.arguments!r}。请重新输出合法的 JSON。"
                        )
                        self.logger.error(f"tool {name} 参数 JSON 解析失败: {e!r}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": err,
                        })
                        if use_mem:
                            self.memory.add_tool_message(tool_call.id, err)
                        tracer.log_tool(step, name, {}, err, 0.0, ok=False, error=str(e))
                        continue

                    # --- 重复调用检测 ---
                    fp = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                    call_fingerprints[fp] = call_fingerprints.get(fp, 0) + 1
                    if call_fingerprints[fp] > MAX_REPEAT_CALLS:
                        hint = (
                            f"注意：你已经用完全相同的参数调用过 {name} {call_fingerprints[fp]} 次了，"
                            f"结果不会变化。请基于已有结果继续，或换用其他工具。"
                        )
                        self.logger.warning(f"检测到重复调用: {fp}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": hint,
                        })
                        if use_mem:
                            self.memory.add_tool_message(tool_call.id, hint)
                        tracer.log_tool(step, name, args, hint, 0.0, ok=False,
                                        error="duplicate_call")
                        continue

                    # --- 执行工具 ---
                    print(f"调用工具: {name}({args})")
                    t0 = time.time()
                    result, ok = self.act(name, args)
                    latency = time.time() - t0
                    print(f"执行结果: {result}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
                    if use_mem:
                        self.memory.add_tool_message(tool_call.id, result)
                    tracer.log_tool(step, name, args, result, latency, ok=ok)

            # 轮数耗尽
            answer = "抱歉，这个问题需要的步骤超出了处理上限，我没能得出最终答案。请把问题拆得更具体一些，或者补充一些信息。"
            if use_mem:
                self.memory.add_assistant_message(answer)
            tracer.finish(answer, terminated_by="max_steps")
            self.logger.warning(f"达到最大轮数 {self.max_steps}: {question}")
            return answer

        except Exception as e:
            tracer.finish("", terminated_by="error", error=repr(e))
            self.logger.error(f"Agent 运行异常: {e!r}")
            raise

    # ---------- 记忆管理 ----------

    def clear_memory(self):
        """清空对话记忆"""
        self.memory.clear()

    def set_memory_summarizer(self, fn):
        """
        注入摘要器，用于把超出窗口的旧对话压成摘要。

        用法：
            from rag.chain import RAGChain
            agent.set_memory_summarizer(lambda msgs: "某段摘要")
        """
        self.memory.set_summarizer(fn)
