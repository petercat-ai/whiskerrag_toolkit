import hashlib
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
)

from whiskerrag_types.model.splitter import (
    BaseSplitConfig,
    JSONSplitConfig,
    MarkdownSplitConfig,
    PDFSplitConfig,
    TextSplitConfig,
)


class MetadataSerializer:
    @staticmethod
    def deep_sort_dict(data: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
        if isinstance(data, dict):
            return {
                k: MetadataSerializer.deep_sort_dict(data[k])
                for k in sorted(data.keys())
            }
        elif isinstance(data, list):
            return [MetadataSerializer.deep_sort_dict(item) for item in data]
        return data

    @staticmethod
    @lru_cache(maxsize=1024)
    def serialize(metadata: Optional[Dict]) -> Optional[Dict]:
        if metadata is None:
            return None
        sorted_metadata = MetadataSerializer.deep_sort_dict(metadata)
        return sorted_metadata if isinstance(sorted_metadata, dict) else None


def calculate_sha256(text: str) -> str:
    text_bytes = text.encode("utf-8")
    sha256_hash = hashlib.sha256()
    sha256_hash.update(text_bytes)
    return sha256_hash.hexdigest()


class KnowledgeSourceEnum(str, Enum):
    GITHUB_REPO = "github_repo"
    GITHUB_FILE = "github_file"
    S3 = "S3"
    USER_INPUT_TEXT = "user_input_text"


class GithubRepoSourceConfig(BaseModel):
    repo_name: str = Field(..., description="github repo url")
    branch: Optional[str] = Field(None, description="branch name of the repo")
    commit_id: Optional[str] = Field(None, description="commit id of the repo")
    auth_info: Optional[str] = Field(None, description="authentication information")


class GithubFileSourceConfig(GithubRepoSourceConfig):
    path: str = Field(..., description="path of the file in the repo")


class S3SourceConfig(BaseModel):
    bucket: str = Field(..., description="s3 bucket name")
    key: str = Field(..., description="s3 key")
    version_id: Optional[str] = Field(None, description="s3 version id")
    region: Optional[str] = Field(None, description="s3 region")
    access_key: Optional[str] = Field(None, description="s3 access key")
    secret_key: Optional[str] = Field(None, description="s3 secret key")
    session_token: Optional[str] = Field(None, description="s3 session token")


class TextSourceConfig(BaseModel):
    text: str = Field(
        default="",
        min_length=1,
        max_length=100000,
        description="Text content, length range 1-100000 characters",
    )


class KnowledgeTypeEnum(str, Enum):
    """
    mime type of the knowledge
    """

    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    PDF = "pdf"
    CSV = "csv"
    DOCX = "docx"
    PPTX = "pptx"
    FOLDER = "folder"
    QA = "qa"


class EmbeddingModelEnum(str, Enum):
    OPENAI = "openai"
    QWEN = "qwen"


KnowledgeSplitConfig = Union[
    BaseSplitConfig,
    MarkdownSplitConfig,
    PDFSplitConfig,
    TextSplitConfig,
    JSONSplitConfig,
]


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
        auth_info (Optional[str]): Authentication information.
        embedding_model_name (Optional[str]): Name of the embedding model.
        metadata (Optional[dict]): Additional metadata.
    """

    space_id: str = Field(
        ...,
        description="the space of knowledge, example: petercat bot id, github repo name",
    )
    knowledge_type: KnowledgeTypeEnum = Field(
        KnowledgeTypeEnum.TEXT, description="type of knowledge resource"
    )
    knowledge_name: str = Field(
        ..., max_length=255, description="name of the knowledge resource"
    )
    source_type: KnowledgeSourceEnum = Field(
        KnowledgeSourceEnum.USER_INPUT_TEXT, description="source type"
    )
    source_config: Union[
        GithubRepoSourceConfig, GithubFileSourceConfig, S3SourceConfig, TextSourceConfig
    ] = Field(
        ...,
        description="source config of the knowledge",
    )
    embedding_model_name: EmbeddingModelEnum = Field(
        EmbeddingModelEnum.OPENAI, description="name of the embedding model"
    )
    split_config: KnowledgeSplitConfig = Field(
        ...,
        description="configuration for splitting the knowledge",
    )
    file_sha: Optional[str] = Field(None, description="SHA of the file")
    file_size: Optional[int] = Field(None, description="size of the file")
    metadata: dict = Field({}, description="additional metadata, user can update it")
    parent_id: Optional[str] = Field(None, description="parent knowledge id")
    enabled: bool = Field(True, description="is knowledge enabled")

    @field_serializer("metadata")
    def serialize_metadata(self, metadata: dict) -> Optional[dict]:
        if metadata is None:
            return None
        sorted_metadata = MetadataSerializer.deep_sort_dict(metadata)
        return sorted_metadata if isinstance(sorted_metadata, dict) else None

    @field_serializer("knowledge_type")
    def serialize_knowledge_type(
        self, knowledge_type: Union[KnowledgeTypeEnum, str]
    ) -> str:
        if isinstance(knowledge_type, KnowledgeTypeEnum):
            return knowledge_type.value
        return str(knowledge_type)

    @field_serializer("source_type")
    def serialize_source_type(
        self, source_type: Union[KnowledgeSourceEnum, str]
    ) -> str:
        if isinstance(source_type, KnowledgeSourceEnum):
            return source_type.value
        return str(source_type)

    @field_serializer("embedding_model_name")
    def serialize_embedding_model_name(
        self, embedding_model_name: Union[EmbeddingModelEnum, str]
    ) -> str:
        if isinstance(embedding_model_name, EmbeddingModelEnum):
            return embedding_model_name.value
        return str(embedding_model_name)

    @field_validator("enabled", mode="before")
    @classmethod
    def convert_tinyint_to_bool(cls, v: Any) -> bool:
        return bool(v)


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

    knowledge_id: str = Field(
        default_factory=lambda: str(uuid4()), description="knowledge id"
    )
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(),
        alias="gmt_create",
        description="creation time",
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(),
        alias="gmt_modified",
        description="update time",
    )
    tenant_id: str = Field(..., description="tenant id")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if (
            self.source_type == KnowledgeSourceEnum.USER_INPUT_TEXT
            and isinstance(self.source_config, TextSourceConfig)
            and self.source_config.text is not None
            and self.file_sha is None
        ):
            self.file_sha = calculate_sha256(self.source_config.text)
            self.file_size = len(self.source_config.text.encode("utf-8"))

    class Config:
        allow_population_by_field_name = True

    @field_serializer("created_at")
    def serialize_created_at(self, created_at: Optional[datetime]) -> Optional[str]:
        return created_at.isoformat() if created_at else None

    @field_serializer("updated_at")
    def serialize_updated_at(self, updated_at: Optional[datetime]) -> Optional[str]:
        return updated_at.isoformat() if updated_at else None

    def update(self, **kwargs: Dict[str, Any]) -> "Knowledge":
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.updated_at = datetime.now()
        return self
