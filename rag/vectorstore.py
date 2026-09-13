"""
向量数据库管理模块
用ChromaDB存储和检索文档向量
"""
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from config import config


class VectorStoreManager:
    """向量数据库管理器"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.vectorstore: Optional[Chroma] = None

    def create_from_documents(self, documents: List[Document]) -> None:
        """从文档列表创建向量数据库"""
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
            )
        else:
            self.vectorstore.add_documents(documents)
        print(f"建库完成: {len(documents)} 个文档")

    def add_documents(self, documents: List[Document]) -> None:
        """往已有数据库添加文档"""
        if self.vectorstore is None:
            print("请先建库")
        else:
            self.vectorstore.add_documents(documents)

    def search(self, query: str, top_k: int = None) -> List[Document]:
        """相似度搜索"""
        k = top_k or config.top_k
        return self.vectorstore.similarity_search(query, k=k)

    def search_with_score(self, query: str, top_k: int = None) -> List[tuple]:
        """带分数的相似度搜索"""
        k = top_k or config.top_k
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def get_retriever(self, top_k: int = None):
        """获取 retriever（转接头，给LCEL链用）"""
        k = top_k or config.top_k
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def clear(self) -> None:
        """清空数据库"""
        self.vectorstore = None
