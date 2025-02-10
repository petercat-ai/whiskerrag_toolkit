from typing import List
from pydantic import BaseModel, Field
from whiskerrag_types.model.chunk import Chunk
from whiskerrag_types.model.knowledge import EmbeddingModelEnum


class RetrievalRequestBase(BaseModel):
    question: str = Field(..., description="The query question")
    embedding_model_name: EmbeddingModelEnum = Field(
        ..., description="The name of the embedding model"
    )
    similarity_threshold: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="The similarity threshold, ranging from 0.0 to 1.0.",
    )
    vector_similarity_weight: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="The weight of vector similarity, ranging from 0.0 to 1.0.",
    )
    top: int = Field(1024, ge=1, description="The maximum number of results to return.")
    # aggs: bool = Field(True, description="是否进行聚合")
    # rerank_mdl: Optional[str] = Field(None, description="重排序模型名称")
    # highlight: bool = Field(False, description="是否高亮显示")


class RetrievalBySpaceRequest(RetrievalRequestBase):
    space_id_list: List[str] = Field(..., description="space id list")


class RetrievalByKnowledgeRequest(RetrievalRequestBase):
    knowledge_id_list: List[str] = Field(..., description="knowledge id list")


class RetrievalChunk(Chunk):
    similarity: float = Field(..., description="The similarity of the chunk")
