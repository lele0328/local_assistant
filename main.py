"""
本地隐私智能助手 - 主程序
入口文件，命令行交互界面
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from tools.base import ToolRegistry
from storage.database import Database
from utils.logger import Logger
from utils.helpers import print_banner


def init_rag():
    """初始化RAG模块"""
    loader = DocumentLoader()
    docs = loader.load_directory(config.docs_path)
    splitter = TextSplitter()
    chunks = splitter.split(docs)
    vectorstore = VectorStoreManager()
    vectorstore.create_from_documents(chunks)
    return vectorstore


def init_tools():
    """初始化工具注册器"""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(NotesTool())
    registry.register(FileSearchTool())
    registry.register(WeatherTool())
    return registry


def init_agent(tool_registry):
    """初始化Agent"""
    agent = Agent()
    for name in tool_registry.get_tool_names():
        tool = tool_registry.tools[name]
        agent.register_tool(
            name=tool.name,
            func=lambda t=tool, **args: t.execute(**args),
            description=tool.description,
            params=tool.get_schema()
        )
    return agent


def main():
    """主循环"""
    print_banner()
    logger = Logger()
    db = Database()
    memory = MemoryManager()

    print("正在初始化系统...")
    try:
        vectorstore = init_rag()
        rag_chain = RAGChain(vectorstore)
    except Exception as e:
        print(f"RAG初始化失败: {e}")
        print("将只使用Agent模式")
        rag_chain = None

    tool_registry = init_tools()
    agent = init_agent(tool_registry)

    print("初始化完成！输入 'quit' 退出，'rag' 切换RAG模式，'agent' 切换Agent模式\n")

    mode = "agent"
    while True:
        try:
            user_input = input(f"[{mode}] 你: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("再见！")
                break
            if user_input.lower() == "rag":
                mode = "rag"
                print("切换到RAG模式\n")
                continue
            if user_input.lower() == "agent":
                mode = "agent"
                print("切换到Agent模式\n")
                continue

            if mode == "rag" and rag_chain:
                answer = rag_chain.ask(user_input)
            else:
                answer = agent.run(user_input)

            print(f"助手: {answer}\n")

            db.save_conversation(user_input, answer)

        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break
        except Exception as e:
            print(f"出错: {e}\n")


if __name__ == "__main__":
    main()
