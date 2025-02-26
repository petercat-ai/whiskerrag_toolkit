from typing import List, Optional, Union

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, MarkdownTextSplitter

from whiskerrag_types.interface.embed_interface import BaseEmbedding
from whiskerrag_types.model.chunk import Chunk
from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
    Knowledge,
    KnowledgeTypeEnum,
)
from whiskerrag_utils import RegisterTypeEnum, register


@register(RegisterTypeEnum.EMBEDDING, EmbeddingModelEnum.OPENAI)
class OpenAIEmbedding(BaseEmbedding):
    async def embed_documents(
        self, knowledge: Knowledge, documents: List[Document]
    ) -> List[Chunk]:
        splitter: Optional[Union[CharacterTextSplitter, MarkdownTextSplitter]] = None
        if knowledge.knowledge_type == KnowledgeTypeEnum.TEXT:
            splitter = CharacterTextSplitter(
                chunk_size=knowledge.split_config.chunk_size,
                chunk_overlap=knowledge.split_config.chunk_overlap,
                separator=",".join(
                    x for x in (knowledge.split_config.separators or [])
                ),
            )
        if knowledge.knowledge_type == KnowledgeTypeEnum.MARKDOWN:
            splitter = MarkdownTextSplitter(
                chunk_size=knowledge.split_config.chunk_size,
                chunk_overlap=knowledge.split_config.chunk_overlap,
            )
        if splitter is None:
            raise Exception("not support knowledge type")

        chunks: List[Chunk] = []
        docs = splitter.split_documents(documents)
        embedding_client = OpenAIEmbeddings()
        for doc in docs:
            embedding = embedding_client.embed_query(doc.page_content)
            chunk = Chunk(
                context=doc.page_content,
                metadata={
                    **doc.metadata,
                },
                embedding=embedding,
                knowledge_id=knowledge.knowledge_id,
                embedding_model_name=knowledge.embedding_model_name,
                space_id=knowledge.space_id,
                tenant_id=knowledge.tenant_id,
            )
            chunks.append(chunk)
        return chunks

    async def embed_text(self, text: str, timeout: Optional[int]) -> List[float]:
        embedding_client = OpenAIEmbeddings(timeout=timeout or 15)
        return embedding_client.embed_query(text)
