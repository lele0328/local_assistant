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
    max_steps: int = Field(default=5, ge=1, le=20, description="Agent最大循环次数")
    system_prompt: str = Field(
        default="你是一个智能助手，可以回答问题、检索文档、做计算。请使用工具完成任务。",
        description="Agent系统提示词"
    )

    # === 存储配置 ===
    db_path: str = Field(default="./data/chroma_db", description="向量数据库路径")
    docs_path: str = Field(default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs"), description="文档目录路径")
    notes_path: str = Field(default="./data/notes.json", description="笔记存储路径")
    log_path: str = Field(default="./data/logs", description="日志路径")


# 全局配置实例，其他模块直接 from config import config 使用
config = AppConfig()
