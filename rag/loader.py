"""
文档加载模块
支持加载 txt、md 格式的文档
"""
from pathlib import Path
from typing import List
from langchain_core.documents import Document


class DocumentLoader:
    """文档加载器，支持多格式文档"""

    SUPPORTED_FORMATS = [".txt", ".md"]

    def __init__(self):
        self.documents: List[Document] = []

    def load_file(self, file_path: str) -> List[Document]:
        """加载单个文件"""
        suffix = Path(file_path).suffix
        if suffix not in self.SUPPORTED_FORMATS:
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        doc = Document(page_content=text, metadata={"source": file_path})
        return [doc]

    def load_directory(self, dir_path: str) -> List[Document]:
        """加载目录下所有支持的文档"""
        for f in Path(dir_path).iterdir():
            if f.is_file() and f.suffix in self.SUPPORTED_FORMATS:
                docs = self.load_file(str(f))
                self.documents.extend(docs)
        return self.documents

    def get_document_count(self) -> int:
        """返回已加载文档数量"""
        return len(self.documents)
