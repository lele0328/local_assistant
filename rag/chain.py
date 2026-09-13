"""
LCEL检索链模块
用LangChain LCEL管道串联：检索 → 格式化 → Prompt → LLM → 输出
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from config import config


# RAG 的 Prompt 模板
RAG_PROMPT_TEMPLATE = """
请根据以下参考资料回答用户的问题。如果资料中没有相关信息，请说"资料中未找到相关内容"。

参考资料：
{context}

用户问题：{question}

回答：
"""


def format_docs(docs):
    """把 Document 列表格式化为纯文本"""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGChain:
    """LCEL 检索链"""

    def __init__(self, vectorstore_manager):
        self.llm = ChatOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model_id,
            temperature=config.temperature,
            timeout=config.timeout,
        )
        self.vectorstore_manager = vectorstore_manager
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    def build_chain(self):
        """构建 LCEL 管道"""
        retriever = self.vectorstore_manager.get_retriever()
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def ask(self, question: str) -> str:
        """提问并获取回答"""
        chain = self.build_chain()
        answer = chain.invoke(question)
        return answer
