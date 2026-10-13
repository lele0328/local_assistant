"""
文本切分模块
把长文档切分成小块，方便向量化和检索
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from config import config


class TextSplitter:
    """文本切分器"""

    def __init__(self):
        Headers_to_split_on = [
            ("#","Header 1"),
            ("##","Header 2"),
            ("###","Header 3")
        ]
        self.splitter =MarkdownHeaderTextSplitter(
            headers_to_split_on=Headers_to_split_on,
            return_each_line=False,
            strip_headers=False
        )

    def split(self, documents: list[Document]) -> List[Document]:
        """把文档列表切分成小块"""
        all_chunks = []
        print(f"切分前: {len(documents)} 个文档")
        for doc in documents:
            chunks = self.splitter.split_text(doc.page_content)
            for chunk in chunks:
                chunk.metadata={**doc.metadata,**chunk.metadata}
            all_chunks.extend(chunks)
        return all_chunks

