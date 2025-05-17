from typing import Generic, List, Optional, TypeVar, Union

from deprecated import deprecated
from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from whiskerrag_types.model.chunk import Chunk
from whiskerrag_types.model.knowledge import EmbeddingModelEnum


@deprecated("RetrievalConfig is deprecated, please use RetrievalRequest instead.")
class RetrievalBaseConfig(BaseModel):
    embedding_model_name: Union[EmbeddingModelEnum, str] = Field(
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

    @field_serializer("embedding_model_name")
    def serialize_embedding_model_name(
        self, embedding_model_name: Optional[Union[EmbeddingModelEnum, str]]
    ) -> Optional[str]:
        if embedding_model_name:
            if isinstance(embedding_model_name, EmbeddingModelEnum):
                return embedding_model_name.value
            else:
                return embedding_model_name
        else:
            return None


@deprecated(
    "RetrievalBySpaceRequest is deprecated, please use RetrievalRequest instead."
)
class RetrievalBySpaceRequest(RetrievalBaseConfig):
    question: str = Field(..., description="The query question")
    space_id_list: List[str] = Field(..., description="space id list")


@deprecated(
    "RetrievalBySpaceRequest is deprecated, please use RetrievalRequest instead."
)
class RetrievalByKnowledgeRequest(RetrievalBaseConfig):
    question: str = Field(..., description="The query question")
    knowledge_id_list: List[str] = Field(..., description="knowledge id list")


class RetrievalConfig(BaseModel):
    type: str = Field(
        ...,
        description="The retrieval type. Each retrieval type corresponds to a specific retriever.",
    )


TConfig = TypeVar("TConfig", bound="RetrievalConfig")


class RetrievalRequest(BaseModel, Generic[TConfig]):
    content: str = Field(
        ...,
        description="The content to be searched, such as a question, text, or any other query string.",
    )
    image_url: Optional[str] = Field(default=None, description="image")
    config: TConfig = Field(
        ...,
        description="The configuration for the retrieval request. Must inherit from RetrievalConfig and have a 'type' attribute.",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "RetrievalRequest":
        if (
            not isinstance(self.config, RetrievalConfig)
            or type(self.config) is RetrievalConfig
        ):
            raise ValueError("config must be a subclass of RetrievalConfig")
        return self

    @field_validator("content")
    def content_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v


class RetrievalChunk(Chunk):
    similarity: float = Field(
        ..., description="The similarity of the chunk, ranging from 0.0 to 1.0."
    )
