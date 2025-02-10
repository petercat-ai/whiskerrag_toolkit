from typing import List, Dict

from whiskerrag_types.interface.retriever_interface import BaseRetriever
from whiskerrag_types.model.retrieval import RetrievalChunk
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.RETRIEVER, "simple")
class SimpleRetriever(BaseRetriever):
    chunk_list: List[RetrievalChunk]
    chunk_index: Dict[str, RetrievalChunk]

    def __init__(self, chunk_list: List[RetrievalChunk]):
        self.chunk_list = chunk_list
        self.chunk_index = self._build_index()

    def _build_index(self) -> Dict[str, str]:
        return {chunk.chunk_id: chunk for chunk in self.chunk_list}

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        results = []
        for doc_id, content in self.doc_index.items():
            if query.lower() in content.lower():
                results.append(
                    {
                        "id": doc_id,
                        "content": content,
                        "score": 1.0,
                    }
                )

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    pass
