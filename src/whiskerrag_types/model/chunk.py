from dataclasses import Field
from datetime import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, field_serializer, Field, field_validator

from .knowledge import EmbeddingModelEnum


class Chunk(BaseModel):
    chunk_id: Optional[str] = Field(None, description="chunk id")
    embedding: Optional[list[float]] = Field(None, description="chunk embedding")
    context: str = Field(..., description="chunk content")
    knowledge_id: str = Field(None, description="file source info")
    embedding_model_name: Optional[EmbeddingModelEnum] = Field(
        EmbeddingModelEnum.OPENAI, description="name of the embedding model"
    )
    space_id: str = Field(..., description="space id")
    metadata: Optional[dict] = Field(None, description="metadata")
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(), description="creation time"
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(), description="update time"
    )

    @field_validator("embedding", mode="before")
    @classmethod
    def parse_embedding(cls, v: Union[str, List[float], None]) -> Optional[List[float]]:
        if v is None:
            return None

        if isinstance(v, list):
            return [float(x) for x in v]

        if isinstance(v, str):
            v = v.strip()
            try:
                import json

                return json.loads(v)
            except json.JSONDecodeError:
                try:
                    if v.startswith("[") and v.endswith("]"):
                        v = v[1:-1]
                    return [float(x.strip()) for x in v.split(",") if x.strip()]
                except ValueError:
                    raise ValueError(f"Invalid embedding format: {v}")

        raise ValueError(f"Unsupported embedding type: {type(v)}")

    @field_serializer("created_at")
    def serialize_created_at(self, created_at: Optional[datetime]):
        return created_at.isoformat() if created_at else None

    @field_serializer("updated_at")
    def serialize_updated_at(self, updated_at: Optional[datetime]):
        return updated_at.isoformat() if updated_at else None

    @field_serializer("embedding_model_name")
    def serialize_embedding_model_name(
        self, embedding_model_name: Optional[EmbeddingModelEnum]
    ):
        return embedding_model_name.value if embedding_model_name else None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.updated_at = datetime.now().isoformat()
        return self
