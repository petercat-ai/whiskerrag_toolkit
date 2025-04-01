from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from whiskerrag_types.interface.embed_interface import Image
from whiskerrag_types.model.knowledge import BaseCharSplitConfig
from whiskerrag_types.model.multi_modal import Text

T = TypeVar("T", bound=BaseCharSplitConfig)
R = TypeVar("R", Text, Image)


class BaseSplitter(Generic[T, R], ABC):
    @abstractmethod
    def split(self, content: str, split_config: T) -> List[R]:
        pass

    @abstractmethod
    def batch_split(self, content: List[str], split_config: T) -> List[List[R]]:
        pass
