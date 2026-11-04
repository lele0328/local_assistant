"""MCP 客户端自检脚本（mcp 2.x）

用官方 SDK 的 stdio transport 真正拉起 mcp_server.py，
列出工具、校验 schema、逐个调用，并验证目录逃逸防护。

用法（必须在项目根目录，且用项目 venv 的解释器）：
    .venv\\Scripts\\python.exe mcp_client.py

退出码：0 全部通过；1 有断言失败。
"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = str(Path(__file__).resolve().parent / "mcp_server.py")

# 越界请求在协议层可能表现为 isError=True，也可能被 server 作为普通文本返回，
# 两种都算「已拦截」，所以这里同时接受。
REFUSAL_MARKERS = ("拒绝", "超出允许范围", "只能搜索项目目录")


def preview(text: str, limit: int = 64) -> str:
    one_line = " ".join(text.split())
    return one_line[:limit] + ("…" if len(one_line) > limit else "")


def is_error(result) -> bool:
    """兼容 mcp 1.x 的 isError 与 2.x 的 is_error。"""
    value = getattr(result, "isError", None)
    if value is None:
        value = getattr(result, "is_error", False)
    return bool(value)


def text_of(result) -> str:
    return "".join(getattr(c, "text", "") for c in (result.content or []))


async def main() -> int:
    failures = []
    params = StdioServerParameters(command=sys.executable, args=[SERVER])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(f"已连接 MCP server: {SERVER}\n")

            # ---- 1. 工具清单与 schema ----
            tools = (await session.list_tools()).tools
            print(f"[工具清单] 共 {len(tools)} 个")
            for t in tools:
                schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None) or {}
                print(
                    f"  - {t.name:<12} 参数={list((schema.get('properties') or {}).keys())}"
                    f" 必填={schema.get('required') or []}"
                )

            expected = {"calculator", "notes", "file_search", "weather", "web_search"}
            got = {t.name for t in tools}
            if got != expected:
                failures.append(f"工具集合不符：缺 {expected - got}，多 {got - expected}")

            # ---- 2. 参数枚举约束 ----
            notes_schema = next((getattr(t, "inputSchema", None) or getattr(t, "input_schema", None) or {} for t in tools if t.name == "notes"), {})
            enum = ((notes_schema.get("properties") or {}).get("action") or {}).get("enum")
            print(f"\n[枚举约束] notes.action => {enum}")
            if set(enum or []) != {"create", "list", "read", "delete"}:
                failures.append(f"notes.action 枚举未生效：{enum}")

            # ---- 3. 逐个调用 ----
            calls = [
                ("calculator", {"expression": "3+5*2"}),
                ("calculator", {"expression": "__import__('os').system('echo hacked')"}),
                ("notes", {"action": "create", "title": "mcp-smoke", "content": "由 mcp_client 写入"}),
                ("notes", {"action": "list"}),
                ("file_search", {"keyword": "README", "directory": "."}),
                ("weather", {"city": "北京"}),
                ("web_search", {"query": "Model Context Protocol"}),
            ]
            print("\n[逐个调用]")
            for name, args in calls:
                result = await session.call_tool(name, args)
                body = text_of(result)
                flag = "OK  " if not is_error(result) else "FAIL"
                print(f"  [{flag}] {name}({json.dumps(args, ensure_ascii=False)[:42]}) -> {preview(body)}")
                if is_error(result):
                    failures.append(f"{name} 调用返回错误: {preview(body, 120)}")

            # ---- 4. 目录逃逸防护 ----
            print("\n[目录逃逸防护]")
            for bad in ("..", "C:\\Windows"):
                result = await session.call_tool("file_search", {"keyword": "x", "directory": bad})
                body = text_of(result)
                blocked = is_error(result) or any(m in body for m in REFUSAL_MARKERS)
                print(
                    f"  [{'OK  ' if blocked else 'FAIL'}] directory={bad!r} "
                    f"{'已拦截' if blocked else '未拦截'} -> {preview(body, 60)}"
                )
                if not blocked:
                    failures.append(f"目录逃逸未拦截：{bad}")

    print()
    if failures:
        print("自检失败：")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("自检全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
