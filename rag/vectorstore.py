"""
向量数据库管理模块
用ChromaDB存储和检索文档向量
"""
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import hashlib
import jieba
from langchain_community.retrievers import BM25Retriever
from rag.hybrid import HybridRetriever


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
        self.bm25= None

    def fingerprint (self,documents):
        h=hashlib.md5()
        for doc in documents:
            h.update(doc.page_content.encode())
        return h.hexdigest()

    def create_from_documents(self, documents: List[Document]) -> None:
        """从文档列表创建向量数据库"""
        self.documents = documents
        new_db = self.fingerprint(documents)
        self.vectorstore = Chroma(
            collection_name=config.collection_name,
            persist_directory=config.db_path,
            embedding_function=self.embeddings
            )
        old_db = (self.vectorstore._collection.metadata or {}).get("fingerprint")
        if old_db != new_db:
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name=config.collection_name,
                persist_directory=config.db_path,
                embedding_function=self.embeddings
            )
            self.vectorstore._collection.modify(metadata={"fingerprint": new_db})

        if len(self.vectorstore) == 0:
            self.vectorstore.add_documents(documents)
        self.bm25=BM25Retriever.from_documents(
            documents=self.documents,
            preprocess_func=jieba.lcut,
            k=10
        )



    def add_documents(self, documents: List[Document]) -> None:
        """往已有数据库添加文档"""
        if self.vectorstore is None:
            print("请先建库")
        else:
            self.vectorstore.add_documents(documents)

    def search(self, query: str, top_k: int = None) -> List[Document]:
        """相似度搜索"""
        k = top_k or config.top_k
        if self.bm25 is not None:
            h=HybridRetriever(
                bm25=self.bm25,
                vector=self.vectorstore.as_retriever(search_kwargs={"k":10}),
                recall_k=10,
                k=k
            )
            return h.invoke(query)
        return self.vectorstore.similarity_search(query=query,k=k)


    def search_with_score(self, query: str, top_k: int = None) -> List[tuple]:
        """带分数的相似度搜索"""
        k = top_k or config.top_k

        return self.vectorstore.similarity_search_with_score(query, k=k)

    def get_retriever(self, top_k: int = None):
        """获取 retriever（转接头，给LCEL链用）"""
        k = top_k or config.top_k
        if self.bm25 is not None:
            return HybridRetriever(
                bm25=self.bm25,
                vector=self.vectorstore.as_retriever(search_kwargs={"k": 10}),
                recall_k=10,
                k=k,
            )
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def reset(self):
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
        self.vectorstore=Chroma(
            collection_name=config.collection_name,
            persist_directory=config.db_path,
            embedding_function=self.embeddings
        )
        return self.vectorstore





