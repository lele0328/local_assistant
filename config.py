"""
配置管理模块
用Pydantic BaseModel做配置，好处：类型检查、自动验证、好维护
"""
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class AppConfig(BaseModel):
    """全局配置类 - 所有配置集中管理"""

    # === LLM 配置 ===
    api_key: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    base_url: str = Field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    model_id: str = Field(default_factory=lambda: os.getenv("LLM_MODEL_ID", ""))
    temperature: float = Field(default=0.3, description="LLM温度，越低越确定")
    timeout: int = Field(default=30, description="API超时秒数")

    # === RAG 配置 ===
    embedding_model: str = Field(default="embedding-3", description="向量模型名")
    chunk_size: int = Field(default=200, ge=50, le=1000, description="文本切分大小")
    chunk_overlap: int = Field(default=50, ge=0, le=200, description="切分重叠区域")
    top_k: int = Field(default=3, ge=1, le=10, description="检索返回文档数")

    # === Agent 配置 ===
    max_steps: int = Field(default=8, ge=1, le=20, description="Agent最大循环次数")
    memory_max_turns: int = Field(
        default=10, ge=1, le=50,
        description="记忆保留的对话轮数（按轮次而非消息条数，保证 tool_calls 配对完整）"
    )
    system_prompt: str = Field(
        default=(
            "你是一个智能助手，可以回答问题、做计算、查天气、管理笔记、搜索本地文件，"
            "也可以从互联网检索最新信息。\n"
            "\n"
            "可用工具：\n"
            "- calculator: 数学计算（只接受纯数学表达式）\n"
            "- file_search: 按文件名搜索本地文件\n"
            "- notes: 笔记的创建/查看/删除\n"
            "- weather: 查询任意城市天气\n"
            "- web_search: 从互联网检索最新信息\n"
            "\n"
            "工作原则：\n"
            "1. 自主判断是否需要工具。只有当工具能提供你确实缺乏的信息时才调用，"
            "不要把简单问题也绕一圈工具。\n"
            "2. 需要实时信息、最新动态、你知识范围之外的内容时，用 web_search。\n"
            "3. 数学计算一律用 calculator，不要自己心算。\n"
            "4. 需要用户所在位置、时间等未提供的上下文时，先向用户询问，不要假设。\n"
            "5. 工具返回错误时，换一种参数或换一个工具重试；连续失败就如实告知用户"
            "你无法完成，不要编造结果。\n"
            "6. 回答基于工具返回的真实结果，不要虚构未获取到的信息。\n"
            "7. 回答简洁，直接给结论，不要复述工具原始输出。"
        ),
        description="Agent系统提示词"
    )

    # === 存储配置 ===
    db_path: str = Field(default="./data/chroma_db", description="向量数据库路径")
    docs_path: str = Field(default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs"), description="文档目录路径")
    notes_path: str = Field(default="./data/notes.json", description="笔记存储路径")
    log_path: str = Field(default="./data/logs", description="日志路径")
    trace_path: str = Field(default="./data/traces", description="运行轨迹路径")


# 全局配置实例，其他模块直接 from config import config 使用
config = AppConfig()
