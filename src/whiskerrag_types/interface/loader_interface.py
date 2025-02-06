from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseLoader(ABC):
    @abstractmethod
    async def load(self) -> List[Document]:
        pass
