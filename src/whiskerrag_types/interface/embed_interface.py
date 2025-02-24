from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document

from whiskerrag_types.model.chunk import Chunk
from whiskerrag_types.model.knowledge import Knowledge


class BaseEmbedding(ABC):
    @abstractmethod
    async def embed_documents(
        self, knowledge: Knowledge, documents: List[Document]
    ) -> List[Chunk]:
        """
        Embed a list of documents.
        """
        pass

    @abstractmethod
    async def embed_text(self, text: str, timeout: Optional[int]) -> List[float]:
        """
        Embed a text.
        """
        pass
