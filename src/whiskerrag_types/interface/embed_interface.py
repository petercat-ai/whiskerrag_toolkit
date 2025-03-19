from abc import ABC, abstractmethod
from typing import List, Optional

from whiskerrag_types.model.multi_modal import Image, Text


class BaseEmbedding(ABC):
    @abstractmethod
    async def embed_text(self, text: Text, timeout: Optional[int]) -> List[float]:
        pass

    @abstractmethod
    async def embed_image(self, image: Image, timeout: Optional[int]) -> List[float]:
        pass
