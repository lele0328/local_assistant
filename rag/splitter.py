"""
文本切分模块
把长文档切分成小块，方便向量化和检索
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config


class TextSplitter:
    """文本切分器"""

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", ",", " ", ""],
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """把文档列表切分成小块"""
        print(f"切分前: {len(documents)} 个文档")
        chunks = self.splitter.split_documents(documents)
        print(f"切分后: {len(chunks)} 个块")
        return chunks

    def split_text(self, text: str) -> List[str]:
        """把纯文本切分成小块（不带metadata）"""
        return self.splitter.split_text(text)
