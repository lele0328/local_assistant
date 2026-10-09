"""
Agent 运行轨迹记录器（可观测性）

替换/补充 utils/logger.py 的用法：不是替代日志，而是专门记录「一次完整运行」。

为什么需要它（面试考点）：
- 你现在 Agent 是黑盒。出问题只能靠肉眼看控制台，关掉终端就没了。
- 面试官问「你线上出过什么问题、怎么定位的」，如果你能打开一个 trace 文件，
  指着某一次失败的 run 说「上下文在这一步开始失真的，因为工具返回被截断了」，
  这比任何背答案都有说服力。
- 这也是「失败案例可回放」的实现基础。

用法：
    from utils.trace import TraceRecorder

    tracer = TraceRecorder()
    tracer.start("北京天气怎么样")

    tracer.log_think(step=1, messages=messages, response=response)   # 记录每轮 LLM 调用
    tracer.log_tool(step=1, name="weather", args={"city": "北京"},
                    result="北京：晴，温度 18°", latency_s=0.8, ok=True)
    tracer.log_answer("北京今天晴，18度。")
    tracer.finish()      # 落盘

落盘位置：./data/traces/{时间戳}_{run_id}.json
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


# 单个字段存进 trace 的最大长度。超长的工具返回是「上下文被污染」的常见原因，
# 所以这里刻意保留长度信息 + 截断内容，方便你看出来到底是哪一步塞爆了上下文。
MAX_TEXT_LEN = 2000


def _truncate(text: Any) -> Any:
    """截断超长文本，同时保留原始长度，便于诊断上下文膨胀"""
    if not isinstance(text, str):
        return text
    if len(text) <= MAX_TEXT_LEN:
        return text
    return {
        "_truncated": True,
        "_original_len": len(text),
        "_preview": text[:MAX_TEXT_LEN],
    }


@dataclass
class ToolCall:
    step: int
    name: str
    args: Dict[str, Any]
    result: Any
    latency_s: float
    ok: bool
    error: str = ""


@dataclass
class ThinkCall:
    step: int
    prompt_tokens_est: int
    message_count: int
    finish_reason: str = ""
    tool_calls_requested: List[str] = field(default_factory=list)
    content_preview: str = ""


@dataclass
class RunTrace:
    """一次完整运行的轨迹"""
    run_id: str = ""
    question: str = ""
    mode: str = "agent"
    started_at: str = ""
    finished_at: str = ""
    total_latency_s: float = 0.0
    final_answer: str = ""
    steps: int = 0
    terminated_by: str = ""          # "answer" / "max_steps" / "error"
    total_tool_calls: int = 0
    failed_tool_calls: int = 0
    thinks: List[ThinkCall] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    error: str = ""


class TraceRecorder:
    """记录一次 Agent / RAG 运行的完整轨迹并落盘"""

    def __init__(self, trace_dir: str = "./data/traces"):
        self.trace_dir = trace_dir
        os.makedirs(self.trace_dir, exist_ok=True)
        self.trace: Optional[RunTrace] = None
        self._t0: float = 0.0

    # ---------- 生命周期 ----------

    def start(self, question: str, mode: str = "agent") -> RunTrace:
        self._t0 = time.time()
        self.trace = RunTrace(
            run_id=uuid.uuid4().hex[:12],
            question=question,
            mode=mode,
            started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        return self.trace

    def finish(self, answer: str = "", terminated_by: str = "answer", error: str = "") -> str:
        """结束并落盘，返回 trace 文件路径"""
        if self.trace is None:
            return ""

        self.trace.final_answer = _truncate(answer)
        self.trace.terminated_by = terminated_by
        self.trace.error = error
        self.trace.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.trace.total_latency_s = round(time.time() - self._t0, 3)
        self.trace.steps = max((t.step for t in self.trace.thinks), default=0)

        path = os.path.join(
            self.trace_dir,
            f"{self.trace.started_at.replace(':', '').replace('-', '').replace(' ', '_')}"
            f"_{self.trace.run_id}.json",
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.trace), f, ensure_ascii=False, indent=2)
        return path

    # ---------- 记录点 ----------

    def log_think(self, step: int, messages: List[Dict], response) -> None:
        """记录一次 LLM 调用（ReAct 的 Thought 阶段）"""
        if self.trace is None:
            return

        # 估算上下文规模：这是发现「上下文被塞爆」的最直接指标
        char_count = sum(
            len(str(m.get("content") or "")) for m in messages if isinstance(m, dict)
        )

        finish_reason = ""
        content_preview = ""
        tool_names: List[str] = []

        try:
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", "") or ""
            msg = choice.message
            content_preview = (getattr(msg, "content", "") or "")[:200]
            for tc in (getattr(msg, "tool_calls", None) or []):
                tool_names.append(tc.function.name)
        except (AttributeError, IndexError, TypeError):
            pass

        self.trace.thinks.append(ThinkCall(
            step=step,
            prompt_tokens_est=char_count // 2,   # 粗估：中文约 2 字符/token
            message_count=len(messages),
            finish_reason=finish_reason,
            tool_calls_requested=tool_names,
            content_preview=content_preview,
        ))

    def log_tool(self, step: int, name: str, args: Dict, result: Any,
                 latency_s: float, ok: bool = True, error: str = "") -> None:
        """记录一次工具调用（ReAct 的 Action + Observation 阶段）"""
        if self.trace is None:
            return

        self.trace.tool_calls.append(ToolCall(
            step=step,
            name=name,
            args=_truncate(args) if isinstance(args, str) else args,
            result=_truncate(result),
            latency_s=round(latency_s, 3),
            ok=ok,
            error=error,
        ))
        self.trace.total_tool_calls += 1
        if not ok:
            self.trace.failed_tool_calls += 1


# ---------- 分析工具 ----------

def load_trace(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_trace(path: str) -> str:
    """打印一份人类可读的 trace 摘要 —— 面试时可以直接展示"""
    t = load_trace(path)
    lines = [
        "=" * 58,
        f"Run {t['run_id']}   [{t['mode']}]",
        f"问题: {t['question']}",
        "=" * 58,
    ]
    for th in t["thinks"]:
        req = ", ".join(th["tool_calls_requested"]) or "(直接回答)"
        lines.append(
            f"  第{th['step']}轮 think  上下文≈{th['prompt_tokens_est']}tok "
            f"({th['message_count']} msgs)  请求工具: {req}"
        )
    for tc in t["tool_calls"]:
        flag = "OK " if tc["ok"] else "ERR"
        lines.append(
            f"      [{flag}] {tc['name']}({tc['args']})  {tc['latency_s']}s"
        )
    lines += [
        "-" * 58,
        f"总轮数: {t['steps']}   工具调用: {t['total_tool_calls']} "
        f"(失败 {t['failed_tool_calls']})",
        f"总耗时: {t['total_latency_s']}s   结束原因: {t['terminated_by']}",
        f"最终答案: {str(t['final_answer'])[:150]}",
        "=" * 58,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print(summarize_trace(sys.argv[1]))
    else:
        # 自测：造一次假运行
        tr = TraceRecorder(trace_dir="./data/traces")
        tr.start("北京天气怎么样")

        class _FakeTC:
            def __init__(self, n):
                self.function = type("F", (), {"name": n})()

        class _FakeMsg:
            content = None
            tool_calls = [_FakeTC("weather")]

        class _FakeChoice:
            finish_reason = "tool_calls"
            message = _FakeMsg()

        class _FakeResp:
            choices = [_FakeChoice()]

        msgs = [{"role": "system", "content": "你是一个助手"}, {"role": "user", "content": "北京天气怎么样"}]
        tr.log_think(1, msgs, _FakeResp())
        tr.log_tool(1, "weather", {"city": "北京"}, "北京：晴，温度18°", 0.81, ok=True)

        class _FakeMsg2:
            content = "北京今天晴，气温 18 度。"
            tool_calls = None

        class _FakeChoice2:
            finish_reason = "stop"
            message = _FakeMsg2()

        class _FakeResp2:
            choices = [_FakeChoice2()]

        tr.log_think(2, msgs, _FakeResp2())
        path = tr.finish("北京今天晴，气温 18 度。")
        print(f"trace 已写入: {path}\n")
        print(summarize_trace(path))
