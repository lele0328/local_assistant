"""
工具函数
"""
import os
import sys


def ensure_directory(path: str):
    """确保目录存在，不存在就创建"""
    os.makedirs(path, exist_ok=True)


def format_timestamp() -> str:
    """格式化当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 100) -> str:
    """截断文本，超长加省略号"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def print_banner():
    """打印程序横幅"""
    print("""
╔══════════════════════════════════════════╗
║     本地隐私智能助手 v1.0                  ║
║     RAG检索 + Agent工具调用 + 本地存储     ║
║     数据不出本地，保护隐私                  ║
╚══════════════════════════════════════════╝
""")
