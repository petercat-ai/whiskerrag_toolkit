from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

from ..model import Knowledge, Chunk


class BaseEmbedding(ABC):
    @abstractmethod
    async def embed(
        self, knowledge: Knowledge, documents: List[Document]
    ) -> List[Chunk]:
        pass
