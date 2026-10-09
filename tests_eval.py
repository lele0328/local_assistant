"""
RAG / Agent 评测脚手架

目的：把你的项目从「看起来能用」变成「有数据证明能用」。

用法：
    # 1) 只跑检索层评测（不需要 LLM API，先看检索质量）
    python tests_eval.py --layer retrieval

    # 2) 跑端到端应答评测（需要 .env 里的 LLM_API_KEY）
    python tests_eval.py --layer e2e

    # 3) 全部
    python tests_eval.py --layer all

    # 4) 输出详细 bad case 报告
    python tests_eval.py --layer all --report

设计说明（面试会问的）：
- 用例分四类：normal（有答案）/ multi_hop（多跳）/ no_answer（必须拒答）/ format（格式约束）
  这四类的划分本身就是面试考点 —— 它保证评测不只测「答得好不好」，
  还测「不知道的时候会不会乱答」，后者才是 RAG 最容易被拷打的点。
- 指标分开算，不混成一个总分。拒答准确率单独统计，因为它是幻觉的负向指标。
- 不依赖 RAGAS 之类的重框架 —— 手写评测逻辑更适合面试讲清楚「我到底测了什么」。
"""
import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 一、测试集：请按你 docs/ 里真实的知识库内容改写这些用例
# ============================================================
# expected_keywords: 回答中必须命中的关键词（任一命中即算命中该条）
# must_refuse: 该条是否应该拒答（无答案类用例）
# 建议最终扩充到 30-50 条，覆盖四类各占 25% 左右。

TEST_CASES = [
    # ---------- 类 1：正常问答（知识库里明确有答案）----------
    {
        "id": "norm_01",
        "type": "normal",
        "question": "什么是装饰器？",
        "expected_keywords": ["函数", "@"],
        "must_refuse": False,
    },
    {
        "id": "norm_02",
        "type": "normal",
        "question": "RAG 解决了哪些问题？",
        "expected_keywords": ["时效", "私有", "幻觉"],
        "must_refuse": False,
    },
    {
        "id": "norm_03",
        "type": "normal",
        "question": "Transformer 是什么时候提出的？",
        "expected_keywords": ["2017", "Google"],
        "must_refuse": False,
    },
    {
        "id": "norm_04",
        "type": "normal",
        "question": "ReAct 循环包含哪几个环节？",
        "expected_keywords": ["思考", "行动", "观察"],
        "must_refuse": False,
    },
    {
        "id": "norm_05",
        "type": "normal",
        "question": "Function Calling 是怎么让模型选择工具的？",
        "expected_keywords": ["名称", "描述", "参数"],
        "must_refuse": False,
    },

    # ---------- 类 2：多跳问答（需要综合两段以上内容）----------
    {
        "id": "hop_01",
        "type": "multi_hop",
        "question": "Transformer 和 RAG 分别解决了什么问题？",
        "expected_keywords": ["注意力", "检索"],
        "must_refuse": False,
    },
    {
        "id": "hop_02",
        "type": "multi_hop",
        "question": "Agent 用 Function Calling 调用工具，这和 RAG 的检索有什么共同点？",
        "expected_keywords": ["外部", "信息"],
        "must_refuse": False,
    },

    # ---------- 类 3：无答案（知识库里没有，必须拒答）----------
    # 这一类最重要：它测的是幻觉率
    {
        "id": "noans_01",
        "type": "no_answer",
        "question": "请告诉我 2027 年诺贝尔物理学奖得主是谁？",
        "expected_keywords": [],
        "must_refuse": True,
    },
    {
        "id": "noans_02",
        "type": "no_answer",
        "question": "我们公司内部报销流程的第三步是什么？",
        "expected_keywords": [],
        "must_refuse": True,
    },
    {
        "id": "noans_03",
        "type": "no_answer",
        "question": "这份文档里提到的张伟的联系电话是多少？",
        "expected_keywords": [],
        "must_refuse": True,
    },
    {
        "id": "noans_04",
        "type": "no_answer",
        "question": "文档作者的家庭住址在哪里？",
        "expected_keywords": [],
        "must_refuse": True,
    },

    # ---------- 类 4：格式 / 行为约束 ----------
    {
        "id": "fmt_01",
        "type": "format",
        "question": "用一句话解释什么是 RAG。",
        "expected_keywords": ["检索"],
        "must_refuse": False,
        "max_sentences": 2,   # 约束：不能超长
    },
    {
        "id": "fmt_02",
        "type": "format",
        "question": "装饰器的三个常见应用场景，用列表列出。",
        "expected_keywords": ["日志", "缓存", "权限"],
        "must_refuse": False,
    },
]

# 判定「拒答」的句式（命中任一即认为模型在拒答）
REFUSAL_PATTERNS = [
    r"未找到相关",
    r"没有相关",
    r"资料中?未?提及",
    r"无法从",
    r"文档中?没有",
    r"未提及",
    r"不确定",
    r"无法回答",
    r"没有足够",
    r"信息不足",
]


@dataclass
class CaseResult:
    """单条用例的结果"""
    id: str
    type: str
    question: str
    answer: str
    hit: bool = False                # 关键词是否命中
    refused: bool = False            # 模型是否拒答
    correct: bool = False            # 该条是否判定正确
    latency_s: float = 0.0
    retrieved_docs: int = 0
    failure_reason: str = ""         # 失败归因（关键！）


@dataclass
class EvalReport:
    """评测汇总"""
    timestamp: str = ""
    layer: str = ""
    total: int = 0
    # 分类指标
    normal_correct: int = 0
    normal_total: int = 0
    multi_hop_correct: int = 0
    multi_hop_total: int = 0
    refuse_correct: int = 0          # 拒答正确数
    refuse_total: int = 0
    format_correct: int = 0
    format_total: int = 0
    # 统计
    avg_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    hallucination_rate: float = 0.0  # 无答案类里「没拒答」的比例
    cases: List[CaseResult] = field(default_factory=list)

    def summarize(self) -> str:
        def pct(a, b):
            return f"{a}/{b} ({a / b * 100:.0f}%)" if b else "-"

        lines = [
            "=" * 58,
            f"评测报告  layer={self.layer}  {self.timestamp}",
            "=" * 58,
            f"正常问答    {pct(self.normal_correct, self.normal_total)}",
            f"多跳问答    {pct(self.multi_hop_correct, self.multi_hop_total)}",
            f"拒答准确率  {pct(self.refuse_correct, self.refuse_total)}   <-- 幻觉负向指标",
            f"格式约束    {pct(self.format_correct, self.format_total)}",
            "-" * 58,
            f"幻觉率      {self.hallucination_rate * 100:.1f}%  （无答案时未拒答的比例）",
            f"平均延迟    {self.avg_latency_s:.2f}s     P95 延迟  {self.p95_latency_s:.2f}s",
            "-" * 58,
            f"总体正确    {pct(sum(1 for c in self.cases if c.correct), self.total)}",
            "=" * 58,
        ]
        return "\n".join(lines)


# ============================================================
# 二、判定逻辑
# ============================================================

def is_refusal(answer: str) -> bool:
    """判断回答是否是拒答"""
    if not answer:
        return True
    return any(re.search(p, answer) for p in REFUSAL_PATTERNS)


def count_sentences(text: str) -> int:
    """粗略统计句子数（中文按。！？分，英文按 .!? 分）"""
    parts = re.split(r"[。！？\n]|(?<!\d)[.!?](?!\d)", text)
    return len([p for p in parts if p.strip()])


def judge(case: dict, answer: str) -> tuple:
    """
    判定单条用例。

    Returns:
        (correct: bool, refused: bool, hit: bool, failure_reason: str)
    """
    refused = is_refusal(answer)

    # --- 无答案类：只要拒答就算对，只要没拒答就算幻觉 ---
    if case.get("must_refuse"):
        if refused:
            return True, True, False, ""
        return False, False, False, "幻觉：知识库中无此信息，但模型没有拒答"

    # --- 有答案类：如果模型拒答了，属于「漏召」---
    if refused:
        return False, True, False, "漏召：知识库中有答案，但模型拒答了"

    # --- 关键词命中判定 ---
    kws = case.get("expected_keywords") or []
    hit = any(k.lower() in answer.lower() for k in kws)

    # --- 格式约束 ---
    if "max_sentences" in case and count_sentences(answer) > case["max_sentences"]:
        return False, False, hit, f"格式：超出句子数限制（{case['max_sentences']}）"

    if not hit:
        return False, False, False, f"答案错误：未命中任何关键词 {kws}"

    return True, False, True, ""


# ============================================================
# 三、两个评测层面
# ============================================================

def eval_retrieval(report: EvalReport):
    """
    只评测检索层：不调 LLM，只看「该召回的 chunk 有没有被召回」。

    价值：把「检索问题」和「生成问题」分开。
    这是面试里区分「Demo 选手」和「工程选手」的关键 ——
    出了问题先判断是 Retrieval Failure 还是 Generation Failure。
    """
    from config import config
    from rag.loader import DocumentLoader
    from rag.splitter import TextSplitter
    from rag.vectorstore import VectorStoreManager

    print("[检索层] 正在初始化...")
    loader = DocumentLoader()
    docs = loader.load_directory(config.docs_path)
    chunks = TextSplitter().split(docs)
    vs = VectorStoreManager()
    vs.create_from_documents(chunks)
    print(f"[检索层] 库内共 {len(chunks)} 个块\n")

    for case in TEST_CASES:
        if case.get("must_refuse"):
            continue  # 无答案类不适用于检索层评测

        t0 = time.time()
        hits = vs.search(case["question"], top_k=config.top_k)
        latency = time.time() - t0
        retrieved_text = "\n".join(d.page_content for d in hits)

        kws = case.get("expected_keywords") or []
        hit = any(k.lower() in retrieved_text.lower() for k in kws)

        r = CaseResult(
            id=case["id"], type=case["type"], question=case["question"],
            answer=retrieved_text[:200],
            hit=hit, correct=hit,
            latency_s=latency, retrieved_docs=len(hits),
            failure_reason="" if hit else f"检索未召回含 {kws} 的内容",
        )
        report.cases.append(r)
        mark = "OK  " if hit else "MISS"
        print(f"[{mark}] {case['id']}  {case['question'][:36]}  ({len(hits)} docs)")

    _finalize(report)


def eval_e2e(report: EvalReport, mode: str = "rag"):
    """
    端到端评测：调真实 LLM，测最终回答质量。

    mode="rag"   走 RAGChain
    mode="agent" 走 Agent（含工具调用）
    """
    from config import config

    if mode == "rag":
        from rag.loader import DocumentLoader
        from rag.splitter import TextSplitter
        from rag.vectorstore import VectorStoreManager
        from rag.chain import RAGChain

        print("[E2E] 正在初始化 RAG...")
        docs = DocumentLoader().load_directory(config.docs_path)
        chunks = TextSplitter().split(docs)
        vs = VectorStoreManager()
        vs.create_from_documents(chunks)
        responder = RAGChain(vs).ask
    else:
        from core.agent import Agent
        from tools.calculator import CalculatorTool
        from tools.notes import NotesTool
        from tools.file_search import FileSearchTool
        from tools.web_search import WebSearchTool
        from tools.base import ToolRegistry

        print("[E2E] 正在初始化 Agent...")
        registry = ToolRegistry()
        for t in (CalculatorTool(), NotesTool(), FileSearchTool(), WebSearchTool()):
            registry.register(t)
        agent = Agent()
        for name in registry.get_tool_names():
            tool = registry.tools[name]
            agent.register_tool(
                name=tool.name,
                func=lambda t=tool, **a: t.execute(**a),
                description=tool.description,
                params=tool.get_schema(),
            )
        responder = agent.run

    print(f"[E2E] 开始评测 {len(TEST_CASES)} 条用例（mode={mode}）\n")

    for case in TEST_CASES:
        t0 = time.time()
        try:
            answer = responder(case["question"])
        except Exception as e:
            answer = f"[执行异常] {e!r}"
        latency = time.time() - t0

        correct, refused, hit, reason = judge(case, answer)

        r = CaseResult(
            id=case["id"], type=case["type"], question=case["question"],
            answer=answer, hit=hit, refused=refused, correct=correct,
            latency_s=latency, failure_reason=reason,
        )
        report.cases.append(r)

        mark = "OK  " if correct else "FAIL"
        print(f"[{mark}] {case['id']} ({latency:.1f}s) {case['question'][:32]}")
        if not correct:
            print(f"       └─ {reason}")
            print(f"       └─ 回答: {answer[:100]}")

    _finalize(report)


# ============================================================
# 四、汇总与输出
# ============================================================

def _finalize(report: EvalReport):
    report.total = len(report.cases)

    buckets = {
        "normal": ("normal_correct", "normal_total"),
        "multi_hop": ("multi_hop_correct", "multi_hop_total"),
        "no_answer": ("refuse_correct", "refuse_total"),
        "format": ("format_correct", "format_total"),
    }
    for c in report.cases:
        keys = buckets.get(c.type)
        if not keys:
            continue
        setattr(report, keys[1], getattr(report, keys[1]) + 1)
        if c.correct:
            setattr(report, keys[0], getattr(report, keys[0]) + 1)

    lat = sorted(c.latency_s for c in report.cases)
    if lat:
        report.avg_latency_s = sum(lat) / len(lat)
        report.p95_latency_s = lat[min(int(len(lat) * 0.95), len(lat) - 1)]

    # 幻觉率 = 无答案类里没拒答的比例
    noans = [c for c in report.cases if c.type == "no_answer"]
    if noans:
        halluc = sum(1 for c in noans if not c.refused)
        report.hallucination_rate = halluc / len(noans)

    report.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_outputs(report: EvalReport, show_detail: bool):
    os.makedirs("./eval_results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 机器可读结果（便于追踪每次改动的效果）
    path = f"./eval_results/eval_{report.layer}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)

    print("\n" + report.summarize())
    print(f"\n结果已保存: {path}")

    if show_detail:
        failures = [c for c in report.cases if not c.correct]
        if failures:
            print("\n" + "=" * 58)
            print("Bad Case 明细（按类型分组 —— 这就是面试要讲的内容）")
            print("=" * 58)
            by_type = {}
            for c in failures:
                by_type.setdefault(c.failure_reason.split("：")[0], []).append(c)
            for reason, items in by_type.items():
                print(f"\n【{reason}】共 {len(items)} 条")
                for c in items:
                    print(f"  - [{c.id}] {c.question}")
                    print(f"    回答: {c.answer[:120]}")


def main():
    ap = argparse.ArgumentParser(description="local_assistant 评测脚手架")
    ap.add_argument("--layer", default="retrieval",
                    choices=["retrieval", "e2e", "all"],
                    help="retrieval=只测检索(不花token) / e2e=端到端 / all=都跑")
    ap.add_argument("--mode", default="rag", choices=["rag", "agent"],
                    help="e2e 评测走哪个链路")
    ap.add_argument("--report", action="store_true", help="输出 bad case 明细")
    args = ap.parse_args()

    reports = []

    if args.layer in ("retrieval", "all"):
        r = EvalReport(layer="retrieval")
        try:
            eval_retrieval(r)
            reports.append(r)
        except Exception as e:
            print(f"[检索层] 初始化失败，跳过: {e}")

    if args.layer in ("e2e", "all"):
        r = EvalReport(layer=f"e2e_{args.mode}")
        try:
            eval_e2e(r, mode=args.mode)
            reports.append(r)
        except Exception as e:
            print(f"[E2E] 失败: {e}")

    for r in reports:
        save_outputs(r, args.report)

    print("\n提示：把上面的指标填进 README 和简历。")
    print("例如：「自建 30 条评测集，正常问答准确率 X%，拒答准确率 Y%，幻觉率 Z%」")


if __name__ == "__main__":
    main()
