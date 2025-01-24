from dataclasses import Field
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, field_serializer, Field


class ResourceType(str, Enum):
    GITHUB_REPO = ("github_repo",)
    GITHUB_FILE = ("github_file",)
    TEXT = "text"
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"


class KnowledgeSplitConfig(BaseModel):
    split_regex: str = Field(description="split regex")
    chunk_size: int = Field(description="chunk size")
    chunk_overlap: int = Field(description="chunk overlap")


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

    knowledge_type: ResourceType = Field(None, description="type of knowledge resource")
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
    embedding_model_name: Optional[str] = Field(
        None, description="name of the embedding model"
    )
    metadata: Optional[dict] = Field(None, description="additional metadata")

    @field_serializer("knowledge_type")
    def serialize_knowledge_type(self, knowledge_type):
        if isinstance(knowledge_type, ResourceType):
            return knowledge_type.value
        return str(knowledge_type)


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
