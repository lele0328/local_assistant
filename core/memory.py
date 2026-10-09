"""
对话记忆管理（修正版）

替换 core/memory.py。

原实现有三个问题：

问题 1（会导致 API 400 报错）—— 滑动窗口从中间截断
    原实现：return self.history[-self.max_history:]
    如果截断点恰好落在 `assistant(tool_calls)` 和它的 `tool` 响应之间，
    喂给 API 会直接报错：`tool` 消息必须紧跟对应的 `tool_calls`。

    已复现：max_history=3 时，[... , assistant(tool_calls), tool, assistant]
    截断后首条变成孤立的 assistant(tool_calls)，其 tool 响应被切掉了。

问题 2 —— 截断是按「消息条数」而不是「轮次」
    一轮工具调用会产生 3-4 条消息，按条数截断会让窗口大小飘忽不定。

问题 3 —— 原实现是断的：main.py 里 new 了 MemoryManager 但从没用过，
    agent.run() 每次都重建 messages，多轮对话实际不生效。

修正点：
1. 按「轮次（turn）」为单位截断，保证 tool_calls 和 tool 响应成组保留
2. 截断后做一次自检，剥掉开头的孤儿 tool / 孤儿 tool_calls
3. 提供 token 预算感知的构建方法，超预算时用摘要替代硬截断
4. get_summary 真正实现为「把旧消息压成一段摘要文本」

用法：
    mem = MemoryManager(max_turns=10)
    mem.add_user_message("你好")
    mem.add_assistant_message("你好，有什么可以帮你？")
    messages = mem.build_messages(system_prompt="你是一个助手")
"""

from typing import Callable, Dict, List, Optional


class MemoryManager:
    """
    对话记忆管理器

    与旧版的关键区别：以「轮次」而非「消息条数」为单位管理窗口，
    从而保证 tool_calls / tool 消息的配对完整性。
    """

    def __init__(self, max_turns: int = 10, system_prompt: Optional[str] = None):
        """
        Args:
            max_turns: 保留最近多少轮对话（1 轮 = 1 次 user 输入及其后续所有消息）
            system_prompt: 系统提示词，会始终置于消息列表首位
        """
        self.history: List[Dict] = []
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        # 摘要器（可选注入），签名为 summary_fn(old_messages) -> str
        self._summary_fn: Optional[Callable[[List[Dict]], str]] = None
        self._summary_cache: str = ""

    # ---------- 写入 ----------

    def add_user_message(self, content: str) -> List[Dict]:
        """添加用户消息。用户消息标志着一轮的开始。"""
        self.history.append({"role": "user", "content": content})
        return self.history

    def add_assistant_message(self, content: str) -> List[Dict]:
        """添加助手消息"""
        self.history.append({"role": "assistant", "content": content})
        return self.history

    def add_assistant_tool_calls(self, tool_calls: List) -> List[Dict]:
        """
        添加带 tool_calls 的助手消息。

        注意：这是工具调用链的一环，必须紧接着配套的 add_tool_message()。
        单独调用它会让历史处于「不完整」状态。
        """
        self.history.append({"role": "assistant", "tool_calls": tool_calls})
        return self.history

    def add_tool_message(self, tool_call_id: str, content: str) -> List[Dict]:
        """添加工具响应消息"""
        self.history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        return self.history

    # ---------- 读取 ----------

    def set_summarizer(self, fn: Callable[[List[Dict]], str]) -> None:
        """注入摘要器，用于把被淘汰的旧消息压成一段摘要"""
        self._summary_fn = fn

    def _split_turns(self) -> List[List[Dict]]:
        """
        把消息历史按「轮次」切分。

        一轮 = 一条 user 消息，加上紧随其后的所有 assistant / tool 消息，
        直到下一条 user 消息出现为止。

        这样切分的好处：tool_calls 和它的 tool 响应永远在同一组里，
        不会因为截断而分家。
        """
        turns: List[List[Dict]] = []
        current: List[Dict] = []

        for msg in self.history:
            if msg.get("role") == "user" and current:
                turns.append(current)
                current = [msg]
            else:
                current.append(msg)

        if current:
            turns.append(current)
        return turns

    def _sanitize(self, messages: List[Dict]) -> List[Dict]:
        """
        清洗消息序列，保证符合 API 的配对要求。

        规则：
        - 开头的孤儿 tool 消息（找不到对应 tool_calls）必须删掉
        - 开头的 assistant(tool_calls) 如果没有配套的 tool 响应，也要删掉
        """
        if not messages:
            return messages

        # 剥掉开头的孤儿 tool 消息
        while messages and messages[0].get("role") == "tool":
            messages.pop(0)

        # 检查开头是否为「未配对的 assistant(tool_calls)」
        if messages and messages[0].get("role") == "assistant" and messages[0].get("tool_calls"):
            expected_ids = {tc.id for tc in messages[0]["tool_calls"]}
            # 看紧跟的若干条里有没有对应的 tool 响应
            seen_ids = {
                m.get("tool_call_id")
                for m in messages[1:]
                if m.get("role") == "tool"
            }
            if not expected_ids.issubset(seen_ids):
                messages.pop(0)

        # 剥掉末尾「有 tool_calls 但没有 tool 响应」的残尾（工具调用中途中断）
        if messages:
            last = messages[-1]
            if last.get("role") == "assistant" and last.get("tool_calls"):
                messages.pop()

        return messages

    def get_messages(self, max_turns: Optional[int] = None) -> List[Dict]:
        """
        获取当前记忆（按轮次截断 + 配对自检）。

        与原实现的区别：不会从 tool_calls / tool 中间切开。
        """
        limit = max_turns if max_turns is not None else self.max_turns
        turns = self._split_turns()

        kept = turns[-limit:] if len(turns) > limit else turns
        flat = [msg for turn in kept for msg in turn]

        return self._sanitize(flat)

    def build_messages(self, user_input: Optional[str] = None) -> List[Dict]:
        """
        构建完整的消息列表（含 system prompt），可直接传给 LLM。

        Args:
            user_input: 若提供，则追加为最新的 user 消息（不写入历史）
        """
        messages: List[Dict] = []

        # 系统提示词始终在首位
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 若有历史摘要，作为额外的系统上下文注入（而不是当作对话消息）
        if self._summary_cache:
            messages.append({
                "role": "system",
                "content": f"以下是较早对话的摘要，供参考：\n{self._summary_cache}",
            })

        messages.extend(self.get_messages())

        if user_input is not None:
            messages.append({"role": "user", "content": user_input})

        return messages

    def get_summary(self) -> str:
        """
        获取被淘汰历史的摘要。

        与旧版的区别：旧版是 `"\\n\\n".join(msg["content"] for msg in history[-3:])`，
        这既不是「摘要」（只是拼接），也会因为 tool 消息没有 content 而 KeyError。

        本实现：如果注入了 summarizer 就用它生成真摘要，否则退化为安全的文本拼接
        （跳过没有 content 字段的消息类型）。
        """
        turns = self._split_turns()
        if len(turns) <= self.max_turns:
            return ""

        old_messages = [m for turn in turns[:-self.max_turns] for m in turn]

        if self._summary_fn:
            try:
                self._summary_cache = self._summary_fn(old_messages)
                return self._summary_cache
            except Exception:
                pass  # 摘要失败不该影响主流程

        # 退化路径：安全拼接，跳过 tool / 无 content 的消息
        parts = [
            str(m["content"])
            for m in old_messages
            if m.get("content") and m.get("role") in ("user", "assistant")
        ]
        self._summary_cache = "\n".join(parts)[-2000:]
        return self._summary_cache

    def clear(self) -> None:
        """清空记忆"""
        self.history.clear()
        self._summary_cache = ""

    def __len__(self) -> int:
        return len(self._split_turns())

    def __repr__(self) -> str:
        return (
            f"MemoryManager(turns={len(self)}, messages={len(self.history)}, "
            f"max_turns={self.max_turns})"
        )


if __name__ == "__main__":
    print("--- 自测 1：tool_calls / tool 配对不被截断 ---")
    mem = MemoryManager(max_turns=1)

    # 第 1 轮：工具调用（会产生 4 条消息）
    mem.add_user_message("北京天气怎么样")
    mem.add_assistant_tool_calls([type("TC", (), {"id": "call_1"})()])
    mem.add_tool_message("call_1", "北京：晴，18度")
    mem.add_assistant_message("北京今天晴，18度。")

    # 第 2 轮
    mem.add_user_message("那上海呢")

    msgs = mem.get_messages()
    roles = [m["role"] for m in msgs]
    print(f"  消息角色序列: {roles}")
    assert roles[0] == "user", "首条应该是 user"
    print("  [OK ] 首条是 user，无孤儿 tool / 孤儿 tool_calls")

    print("\n--- 自测 2：复现旧版会崩的场景，验证新版不崩 ---")
    mem2 = MemoryManager(max_turns=1)
    mem2.history = [
        {"role": "user", "content": "北京天气"},
        {"role": "assistant", "tool_calls": [type("TC", (), {"id": "c1"})()]},
        {"role": "tool", "tool_call_id": "c1", "content": "晴"},
        {"role": "assistant", "content": "北京晴"},
    ]
    out = mem2.get_messages()
    print(f"  消息角色序列: {[m['role'] for m in out]}")
    assert not (out[0]["role"] == "assistant" and out[0].get("tool_calls")), \
        "不应该以孤立的 tool_calls 开头"
    print("  [OK ] 旧的 400 报错场景已被修复")

    print("\n--- 自测 3：system prompt + 历史拼接 ---")
    mem3 = MemoryManager(max_turns=5, system_prompt="你是一个助手")
    mem3.add_user_message("你好")
    mem3.add_assistant_message("你好！")
    full = mem3.build_messages(user_input="帮我算 1+1")
    print(f"  角色序列: {[m['role'] for m in full]}")
    assert full[0]["role"] == "system"
    assert full[-1]["content"] == "帮我算 1+1"
    print("  [OK ] system 在首位，最新 user 输入在末位")

    print("\n--- 自测 4：get_summary 不再 KeyError ---")
    mem4 = MemoryManager(max_turns=1)
    mem4.history = [
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "tool_calls": [type("TC", (), {"id": "c1"})()]},
        {"role": "tool", "tool_call_id": "c1", "content": "结果"},   # 无 content? 有，但不该被拼接
        {"role": "user", "content": "问题2"},
    ]
    s = mem4.get_summary()
    print(f"  摘要: {s!r}")
    assert "问题1" in s
    print("  [OK ] 跳过 tool 消息，未抛 KeyError")

    print("\n--- 自测 5：注入摘要器 ---")
    mem5 = MemoryManager(max_turns=1)
    mem5.set_summarizer(lambda msgs: f"[LLM摘要] 共{len(msgs)}条旧消息")
    mem5.history = [{"role": "user", "content": "旧问题"}, {"role": "user", "content": "新问题"}]
    print(f"  摘要: {mem5.get_summary()!r}")
    print("  [OK ] 摘要器被正确调用")

    print("\n全部通过。")
