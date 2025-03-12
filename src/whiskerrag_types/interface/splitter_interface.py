from abc import ABC, abstractmethod
from typing import List


class BaseSplitter(ABC):
    @abstractmethod
    def split(self, content: str) -> List[str]:
        pass