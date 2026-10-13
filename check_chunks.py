"""
切块体检工具 —— 看你的文档被切成了什么样。

喂给 RAG 的就是这些块：它们会被 embedding 后存进向量库，
检索时命中的就是其中的某几块。所以块切得好不好，直接决定检索质量。

用法（在项目根目录执行）：
    .venv/Scripts/python.exe check_chunks.py
    .venv/Scripts/python.exe check_chunks.py --size 500
    .venv/Scripts/python.exe check_chunks.py --size 500 --overlap 100
    .venv/Scripts/python.exe check_chunks.py --full      # 打印每块完整内容
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from rag.loader import DocumentLoader
from rag.splitter import TextSplitter


def analyze(chunks):
    """统计切块质量。返回一个 dict。"""
    lens = [len(c.page_content) for c in chunks]

    multi_section = []   # 一个块里含两个以上标题 = 跨小节混杂
    no_heading = []      # 不以标题开头的块 = 延续上一块，上下文可能是残缺的
    tiny = []            # 过短的块 = 向量语义过弱，基本检索不到

    for i, c in enumerate(chunks):
        text = c.page_content
        n_head = text.count("##")
        if n_head >= 2:
            multi_section.append(i)
        if not text.lstrip().startswith("#"):
            no_heading.append(i)
        if len(text) < 60:
            tiny.append(i)

    return {
        "count": len(chunks),
        "min": min(lens) if lens else 0,
        "max": max(lens) if lens else 0,
        "avg": sum(lens) / len(lens) if lens else 0,
        "multi_section": multi_section,
        "no_heading": no_heading,
        "tiny": tiny,
    }


def main():
    ap = argparse.ArgumentParser(description="切块体检")
    ap.add_argument("--size", type=int, default=None, help="临时覆盖 chunk_size")
    ap.add_argument("--overlap", type=int, default=None, help="临时覆盖 chunk_overlap")
    ap.add_argument("--full", action="store_true", help="打印每块完整内容")
    ap.add_argument("--max-preview", type=int, default=70, help="预览字数")
    args = ap.parse_args()

    if args.size:
        config.chunk_size = args.size
    if args.overlap is not None:
        config.chunk_overlap = args.overlap

    print("=" * 74)
    print(f"切块参数  chunk_size={config.chunk_size}  chunk_overlap={config.chunk_overlap}")
    print(f"文档目录  {config.docs_path}")
    print("=" * 74)

    docs = DocumentLoader().load_directory(config.docs_path)
    chunks = TextSplitter().split(docs)
    stat = analyze(chunks)

    print()
    print("-" * 74)
    for i, c in enumerate(chunks):
        text = c.page_content
        n_head = text.count("##")
        marks = []
        if n_head >= 2:
            marks.append("跨多节")
        if len(text) < 60:
            marks.append("过短")
        if not text.lstrip().startswith("#"):
            marks.append("无标题")

        tag = ("  <<< " + " / ".join(marks)) if marks else ""
        print(f"[{i:02d}] {len(text):4d}字  {n_head}标题{tag}")

        if args.full:
            for line in text.splitlines():
                print(f"       {line}")
        else:
            preview = text.replace("\n", " | ")[:args.max_preview]
            print(f"       {preview}")

    print("-" * 74)
    print()
    print("=" * 74)
    print(f"总块数        {stat['count']}")
    print(f"长度          min {stat['min']} / avg {stat['avg']:.0f} / max {stat['max']}")
    print("-" * 74)
    print(f"跨多节的块    {len(stat['multi_section'])}  {stat['multi_section']}")
    print(f"过短的块(<60) {len(stat['tiny'])}  {stat['tiny']}")
    print(f"不以标题开头  {len(stat['no_heading'])}  {stat['no_heading']}")
    print("=" * 74)
    print()
    print("怎么读这些指标：")
    print("  跨多节的块  —— 一个块里混了两个小节的语义，它的向量「两边都不像」，")
    print("                 检索时要么两边都召不到，要么误召。这是最该修的。")
    print("  过短的块    —— 文本太短，向量语义信号很弱，几乎不可能被正确检索到。")
    print("  无标题开头的块 —— 它延续上一块。如果上一块也没带标题，这块就失去了")
    print("                 「我属于哪个小节」的信息，模型看到它时不知道上下文是什么。")
    print()


if __name__ == "__main__":
    main()
