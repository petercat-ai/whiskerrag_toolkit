from abc import ABC, abstractmethod
from typing import List

from whiskerrag_types.model.knowledge import KnowledgeSplitConfig


class BaseSplitter(ABC):
    @abstractmethod
    def split(self, content: str, config: KnowledgeSplitConfig) -> List[str]:
        pass
