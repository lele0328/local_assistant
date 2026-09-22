"""
本地隐私智能助手 - FastAPI API服务
启动方式：uvicorn demo_api:app --reload
文档地址：http://localhost:8000/docs
"""
import sys
import os
import time
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from config import config
from core.agent import Agent
from core.memory import MemoryManager
from rag.loader import DocumentLoader
from rag.splitter import TextSplitter
from rag.vectorstore import VectorStoreManager
from rag.chain import RAGChain
from tools.calculator import CalculatorTool
from tools.notes import NotesTool
from tools.file_search import FileSearchTool
from tools.weather import WeatherTool
from tools.web_search import WebSearchTool
from tools.base import ToolRegistry
from storage.database import Database
from utils.logger import Logger


# ========== 装饰器 ==========

def log(func):
    """日志装饰器：记录函数调用"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"开始执行: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"执行完毕: {func.__name__}")
        return result
    return wrapper


def timer(func):
    """计时装饰器：测量执行时间"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"耗时: {elapsed:.3f}秒")
        return result
    return wrapper


# ========== 请求/响应模型 ==========

class ChatRequest(BaseModel):
    question: str
    mode: str = "agent"  # agent 或 rag


class ChatResponse(BaseModel):
    answer: str
    mode: str


class WeatherResponse(BaseModel):
    city: str
    weather: str
    temperature: str


# ========== 初始化全局组件 ==========

app = FastAPI(title="本地智能助手API", version="2.0")

logger = Logger()
db = Database()

# 初始化RAG
print("正在初始化RAG...")
try:
    loader = DocumentLoader()
    docs = loader.load_directory(config.docs_path)
    splitter = TextSplitter()
    chunks = splitter.split(docs)
    vectorstore = VectorStoreManager()
    vectorstore.create_from_documents(chunks)
    rag_chain = RAGChain(vectorstore)
    print("RAG初始化成功")
except Exception as e:
    print(f"RAG初始化失败: {e}，将只使用Agent模式")
    rag_chain = None

# 初始化工具和Agent
print("正在初始化Agent...")
tool_registry = ToolRegistry()
tool_registry.register(CalculatorTool())
tool_registry.register(NotesTool())
tool_registry.register(FileSearchTool())
tool_registry.register(WeatherTool())
tool_registry.register(WebSearchTool())

agent = Agent()
for name in tool_registry.get_tool_names():
    tool = tool_registry.tools[name]
    agent.register_tool(
        name=tool.name,
        func=lambda t=tool, **args: t.execute(**args),
        description=tool.description,
        params=tool.get_schema()
    )
print("Agent初始化成功")


# ========== API接口 ==========

@app.get("/")
def root():
    """根路径"""
    return {"message": "本地智能助手API", "docs": "访问 /docs 查看接口文档"}


@app.get("/health")
def health():
    """健康检查"""
    return {"status": "ok", "rag": rag_chain is not None}


@app.post("/chat", response_model=ChatResponse)
@timer
def chat(request: ChatRequest):
    """聊天接口 - 支持agent和rag两种模式"""
    try:
        if request.mode == "rag" and rag_chain:
            answer = rag_chain.ask(request.question)
        else:
            answer = agent.run(request.question)

        db.save_conversation(request.question, answer)
        return ChatResponse(answer=answer, mode=request.mode)

    except Exception as e:
        return ChatResponse(answer=f"出错了: {e}", mode=request.mode)


@app.get("/weather/{city}", response_model=WeatherResponse)
def get_weather(city: str):
    """查询城市天气 - 接上真实WeatherTool"""
    weather_tool = WeatherTool()
    result = weather_tool.execute(city=city)
    return WeatherResponse(city=city, weather=result, temperature="")


@app.get("/search")
def web_search(query: str):
    """网络搜索 - 直接调用WebSearchTool"""
    search_tool = WebSearchTool()
    result = search_tool.execute(query=query)
    return {"query": query, "result": result}


@app.get("/conversations")
def get_conversations(limit: int = 10):
    """获取历史对话"""
    conversations = db.get_conversations(limit=limit)
    return {"count": len(conversations), "conversations": conversations}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文件到RAG知识库（支持 .txt 和 .md）"""
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in [".txt", ".md"]:
        return {"error": "只支持 .txt 和 .md 格式"}

    content = await file.read()
    print(f"[upload] 收到文件: {file.filename}, 大小: {len(content)} bytes")

    # 用 chardet 自动检测编码
    encoding = None
    try:
        import chardet
        detected = chardet.detect(content)
        encoding = detected.get("encoding") if detected.get("confidence", 0) > 0.7 else None
    except ImportError:
        pass

    # 按优先级尝试解码
    text = None
    for enc in [encoding, "utf-8", "gbk", "gb18030", "latin-1"]:
        if not enc:
            continue
        try:
            text = content.decode(enc)
            print(f"[upload] {enc}解码成功, 文本长度: {len(text)}")
            break
        except (UnicodeDecodeError, TypeError):
            continue

    if not text:
        text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        return {"error": "文件内容为空"}

    # 保存文件到 docs 目录
    file_path = os.path.join(config.docs_path, file.filename)
    with open(file_path, "wb") as f:
        f.write(content)

    # 切分并加入向量库
    from langchain_core.documents import Document
    doc = Document(page_content=text, metadata={"source": file.filename})
    sp = TextSplitter()
    chunks = sp.split([doc])
    print(f"[upload] 切分结果: {len(chunks)} 个块")

    if not chunks:
        return {"error": "文件切分后没有内容，请检查文件内容是否有效"}

    if rag_chain:
        rag_chain.vectorstore_manager.add_documents(chunks)
        return {"message": f"文件 '{file.filename}' 已上传并加入知识库", "chunks": len(chunks)}
    else:
        return {"message": f"文件已保存到 docs 目录，但RAG未初始化，重启后生效", "chunks": len(chunks)}


# ========== 启动方式 ==========
# 终端运行：uvicorn demo_api:app --reload
# 然后访问：http://localhost:8000/docs
