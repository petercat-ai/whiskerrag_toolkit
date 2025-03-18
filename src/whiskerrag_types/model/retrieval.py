from typing import List, Optional

from pydantic import BaseModel, Field, field_serializer

from whiskerrag_types.model.chunk import Chunk
from whiskerrag_types.model.knowledge import EmbeddingModelEnum


class RetrievalRequestBase(BaseModel):
    question: str = Field(..., description="The query question")
    embedding_model_name: EmbeddingModelEnum = Field(
        ..., description="The name of the embedding model"
    )
    similarity_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="The similarity threshold, ranging from 0.0 to 1.0.",
    )
    top: int = Field(1024, ge=1, description="The maximum number of results to return.")
    metadata_filter: dict = Field({}, description="metadata filter")
    # aggs: bool = Field(True, description="是否进行聚合")
    # rerank_mdl: Optional[str] = Field(None, description="重排序模型名称")
    # highlight: bool = Field(False, description="是否高亮显示")

    @field_serializer("embedding_model_name")
    def serialize_embedding_model_name(
        self, embedding_model_name: Optional[EmbeddingModelEnum]
    ) -> Optional[str]:
        return embedding_model_name.value if embedding_model_name else None


class RetrievalBySpaceRequest(RetrievalRequestBase):
    space_id_list: List[str] = Field(..., description="space id list")


class RetrievalByKnowledgeRequest(RetrievalRequestBase):
    knowledge_id_list: List[str] = Field(..., description="knowledge id list")


class RetrievalChunk(Chunk):
    similarity: float = Field(..., description="The similarity of the chunk")
