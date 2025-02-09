from typing import List
from langchain_text_splitters import CharacterTextSplitter, MarkdownTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from whiskerrag_types.interface.embed_interface import BaseEmbedding
from whiskerrag_types.model.chunk import Chunk
from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
    Knowledge,
    KnowledgeType,
)
from whiskerrag_utils.registry import register


@register(EmbeddingModelEnum.OPENAI)
class OpenAIEmbedding(BaseEmbedding):
    async def embed(
        self, knowledge: Knowledge, documents: List[Document]
    ) -> List[Chunk]:
        print(f"start embed knowledge: {knowledge}")
        splitter = None
        if knowledge.knowledge_type == KnowledgeType.TEXT:
            splitter = CharacterTextSplitter(
                chunk_size=knowledge.split_config.get("chunk_size"),
                chunk_overlap=knowledge.split_config.get("chunk_overlap"),
            )
        if knowledge.knowledge_type == KnowledgeType.MARKDOWN:
            splitter = MarkdownTextSplitter(
                chunk_size=knowledge.split_config.get("chunk_size"),
                chunk_overlap=knowledge.split_config.get("chunk_overlap"),
            )
        if splitter is None:
            raise Exception("not support knowledge type")

        chunks: List[Chunk] = []
        docs = splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        for doc in docs:
            print(f"doc: {doc.page_content}")
            embedding = embeddings.embed_query(doc.page_content)
            print(f"embedding result: {embedding}")
            chunk = Chunk(
                context=doc.page_content,
                metadata={
                    **doc.metadata,
                },
                embedding=embedding,
                knowledge_id=knowledge.knowledge_id,
                embedding_model_name=knowledge.embedding_model_name,
                space_id=knowledge.space_id,
            )
            chunks.append(chunk)
        return chunks
