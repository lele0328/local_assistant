from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import Any


class HybridRetriever(BaseRetriever):
    bm25 : Any=None
    vector : Any=None
    recall_k: int=10
    k : int=3

    def _get_relevant_documents(self, query: str) -> list[Document]:
        score={}
        pool={}
        bm25_result=self.bm25.invoke(query)[:self.recall_k] if self.bm25 else []
        vector_result=self.vector.invoke(query)[:self.recall_k] if self.vector else[]
        if not bm25_result:
            return vector_result
        if not vector_result :
            return bm25_result
        rank_lists=[vector_result,bm25_result]
        for rank_list in rank_lists:
            for rank,doc in enumerate(rank_list,start=1):
                doc_id=doc.page_content
                score[doc_id]=score.get(doc_id, 0) + 1.0 / (60 + rank)
                pool[doc_id]=doc
        order = sorted(score, key=lambda x: -score[x])[:self.k]
        return [pool[k] for k in order]



