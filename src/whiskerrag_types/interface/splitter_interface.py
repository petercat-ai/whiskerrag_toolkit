from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from whiskerrag_types.model.knowledge import BaseSplitConfig

T = TypeVar("T", bound=BaseSplitConfig)


class BaseSplitter(Generic[T], ABC):
    @abstractmethod
    def split(self, content: str, split_config: T) -> List[str]:
        pass

    @abstractmethod
    def batch_split(self, content: List[str], split_config: T) -> List[List[str]]:
        pass
