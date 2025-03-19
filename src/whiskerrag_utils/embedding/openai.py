from typing import List, Optional

from langchain_openai import OpenAIEmbeddings

from whiskerrag_types.interface.embed_interface import BaseEmbedding, Image
from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
)
from whiskerrag_utils import RegisterTypeEnum, register


@register(RegisterTypeEnum.EMBEDDING, EmbeddingModelEnum.OPENAI)
class OpenAIEmbedding(BaseEmbedding):
    async def embed_text(self, text: str, timeout: Optional[int]) -> List[float]:
        embedding_client = OpenAIEmbeddings(timeout=timeout or 15)
        embedding = embedding_client.embed_query(text)
        return embedding

    async def embed_image(self, image: Image, timeout: Optional[int]) -> List[float]:
        return []
