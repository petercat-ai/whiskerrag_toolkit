from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from pydantic import BaseModel

from whiskerrag_types.interface.embed_interface import Image
from whiskerrag_types.model.multi_modal import Document, Text

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R", Text, Image, Document)


class BaseParser(Generic[T, R], ABC):
    @abstractmethod
    def split(self, content: str, split_config: T) -> List[R]:
        pass

    @abstractmethod
    def batch_split(self, content: List[str], split_config: T) -> List[List[R]]:
        pass
