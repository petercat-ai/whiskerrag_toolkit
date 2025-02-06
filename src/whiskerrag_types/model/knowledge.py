from dataclasses import Field
from datetime import datetime
from enum import Enum
import hashlib
from typing import List, Optional

from pydantic import BaseModel, field_serializer, Field


def calculate_sha256(text):
    # 将文本转换为 UTF-8 编码的字节
    text_bytes = text.encode("utf-8")

    # 创建 SHA-256 哈希对象
    sha256_hash = hashlib.sha256()

    # 更新哈希对象
    sha256_hash.update(text_bytes)

    # 返回十六进制形式的哈希值
    return sha256_hash.hexdigest()


class KnowledgeSourceType(str, Enum):
    GITHUB_REPO = "github_repo"
    S3 = "S3"
    TEXT = "text"


class KnowledgeType(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    PDF = "pdf"
    CSV = "csv"
    DOCX = "docx"
    PPTX = "pptx"
    IMAGE = "image"


class EmbeddingModelEnum(str, Enum):
    OPENAI = "openai"
    qwen = "qwen"


class KnowledgeSplitConfig(BaseModel):
    separators: List[str] = Field(None, description="separators")
    split_regex: str = Field(description="split regex")
    chunk_size: int = Field(2000, description="chunk size")
    chunk_overlap: int = Field(200, description="chunk overlap")


class KnowledgeCreate(BaseModel):
    """
    KnowledgeCreate model for creating knowledge resources.
    Attributes:
        knowledge_type (ResourceType): Type of knowledge resource.
        space_id (str): Space ID, example: petercat bot ID.
        knowledge_name (str): Name of the knowledge resource.
        file_sha (Optional[str]): SHA of the file.
        file_size (Optional[int]): Size of the file.
        split_config (Optional[dict]): Configuration for splitting the knowledge.
        source_data (Optional[str]): Source data of the knowledge.
        source_url (Optional[str]): URL of the source.
        auth_info (Optional[str]): Authentication information.
        embedding_model_name (Optional[str]): Name of the embedding model.
        metadata (Optional[dict]): Additional metadata.
    """

    source_type: KnowledgeSourceType = Field(description="source type")
    knowledge_type: KnowledgeType = Field(
        None, description="type of knowledge resource"
    )
    space_id: str = Field(None, description="space id, example: petercat bot id")
    knowledge_name: str = Field(None, description="name of the knowledge resource")
    file_sha: Optional[str] = Field(None, description="SHA of the file")
    file_size: Optional[int] = Field(None, description="size of the file")
    split_config: Optional[dict] = Field(
        None, description="configuration for splitting the knowledge"
    )
    source_data: Optional[str] = Field(None, description="source data of the knowledge")
    source_url: Optional[str] = Field(
        None,
        description="URL of the source",
        pattern=r"^(https?|ftp)://[^\s/$.?#].[^\s]*$",
    )
    auth_info: Optional[str] = Field(None, description="authentication information")
    embedding_model_name: Optional[EmbeddingModelEnum] = Field(
        EmbeddingModelEnum.OPENAI, description="name of the embedding model"
    )
    metadata: Optional[dict] = Field(None, description="additional metadata")

    @field_serializer("knowledge_type")
    def serialize_knowledge_type(self, knowledge_type):
        if isinstance(knowledge_type, KnowledgeType):
            return knowledge_type.value
        return str(knowledge_type)

    @field_serializer("source_type")
    def serialize_source_type(self, source_type):
        if isinstance(source_type, KnowledgeSourceType):
            return source_type.value
        return str(source_type)

    @field_serializer("embedding_model_name")
    def serialize_embedding_model_name(self, embedding_model_name):
        if isinstance(embedding_model_name, EmbeddingModelEnum):
            return embedding_model_name.value
        return str(embedding_model_name)


class KnowledgeResponse(BaseModel):
    knowledge_id: Optional[str] = None
    space_id: str = None
    knowledge_name: str
    sha: Optional[str] = None
    source_url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    space_id: str
    tenant_id: str


class Knowledge(KnowledgeCreate):
    """
    Knowledge model class that extends KnowledgeCreate.
    Attributes:
        knowledge_id (str): Knowledge ID.
        tenant_id (str): Tenant ID.
        created_at (Optional[datetime]): Creation time, defaults to current time in ISO format.
        updated_at (Optional[datetime]): Update time, defaults to current time in ISO format.
    Methods:
        serialize_created_at(created_at: Optional[datetime]) -> Optional[str]:
            Serializes the created_at attribute to ISO format.
        serialize_updated_at(updated_at: Optional[datetime]) -> Optional[str]:
            Serializes the updated_at attribute to ISO format.
        update(**kwargs) -> 'Knowledge':
            Updates the attributes of the instance with the provided keyword arguments and sets updated_at to the current time.
    """

    knowledge_id: str = Field(None, description="knowledge id")
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(), description="creation time"
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(), description="update time"
    )
    tenant_id: str = Field(..., description="tenant id")

    def __init__(self, **data):
        super().__init__(**data)
        if (
            self.source_data is not None
            and self.file_sha == None
            and self.source_type == KnowledgeSourceType.TEXT
        ):
            self.file_sha = calculate_sha256(self.source_data)
            self.file_size = len(self.source_data.encode("utf-8"))

    @field_serializer("created_at")
    def serialize_created_at(self, created_at: Optional[datetime]):
        return created_at.isoformat() if created_at else None

    @field_serializer("updated_at")
    def serialize_updated_at(self, updated_at: Optional[datetime]):
        return updated_at.isoformat() if updated_at else None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.updated_at = datetime.now().isoformat()
        return self
