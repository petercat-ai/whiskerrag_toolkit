from typing import Literal, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    model_validator,
)

from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
    GithubRepoSourceConfig,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
    MetadataSerializer,
    OpenIdSourceConfig,
    OpenUrlSourceConfig,
    S3SourceConfig,
    TextSourceConfig,
)
from whiskerrag_types.model.splitter import (
    BaseCharSplitConfig,
    JSONSplitConfig,
    MarkdownSplitConfig,
    PDFSplitConfig,
    TextSplitConfig,
)


class KnowledgeCreateBase(BaseModel):
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
    metadata: dict = Field({}, description="additional metadata, user can update it")
    source_type: KnowledgeSourceEnum = Field(..., description="source type")
    embedding_model_name: Union[EmbeddingModelEnum, str] = Field(
        EmbeddingModelEnum.OPENAI,
        description="name of the embedding model. you can set any other model if target embedding service registered",
    )
    file_sha: Optional[str] = Field(None, description="SHA of the file")
    file_size: Optional[int] = Field(None, description="size of the file")

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


class TextCreate(KnowledgeCreateBase):
    knowledge_type: Literal[KnowledgeTypeEnum.TEXT] = Field(
        KnowledgeTypeEnum.TEXT, description="type of knowledge resource"
    )
    source_config: Union[
        TextSourceConfig, OpenUrlSourceConfig, OpenIdSourceConfig, S3SourceConfig
    ] = Field(
        ...,
        description="source config of the knowledge",
    )
    split_config: TextSplitConfig = Field(
        ...,
        description="split config of the knowledge",
    )


class JSONCreate(KnowledgeCreateBase):
    knowledge_type: Literal[KnowledgeTypeEnum.JSON] = Field(
        KnowledgeTypeEnum.JSON, description="type of knowledge resource"
    )
    source_config: Union[
        TextSourceConfig, OpenUrlSourceConfig, OpenIdSourceConfig, S3SourceConfig
    ] = Field(
        ...,
        description="source config of the knowledge",
    )
    split_config: JSONSplitConfig = Field(
        ...,
        description="split config of the knowledge",
    )


class MarkdownCreate(KnowledgeCreateBase):
    knowledge_type: Literal[KnowledgeTypeEnum.MARKDOWN] = Field(
        KnowledgeTypeEnum.MARKDOWN, description="type of knowledge resource"
    )
    source_config: Union[
        TextSourceConfig, OpenUrlSourceConfig, OpenIdSourceConfig, S3SourceConfig
    ] = Field(
        ...,
        description="source config of the knowledge",
    )
    split_config: MarkdownSplitConfig = Field(
        ...,
        description="split config of the knowledge",
    )


class PDFCreate(KnowledgeCreateBase):
    knowledge_type: Literal[KnowledgeTypeEnum.PDF] = Field(
        KnowledgeTypeEnum.PDF, description="type of knowledge resource"
    )
    source_config: Union[OpenUrlSourceConfig, OpenIdSourceConfig, S3SourceConfig] = (
        Field(
            ...,
            description="source config of the knowledge",
        )
    )
    split_config: PDFSplitConfig = Field(
        ...,
        description="split config of the knowledge",
    )
    file_sha: str = Field(..., description="SHA of the file")
    file_size: int = Field(..., description="Byte size of the file")


class GithubRepoCreate(KnowledgeCreateBase):
    knowledge_type: Literal[KnowledgeTypeEnum.FOLDER] = Field(
        KnowledgeTypeEnum.FOLDER, description="type of knowledge resource"
    )
    source_config: GithubRepoSourceConfig = Field(
        ...,
        description="source config of the knowledge",
    )
    split_config: BaseCharSplitConfig = Field(
        ...,
        description="split config of the knowledge",
    )


class QACreate(KnowledgeCreateBase):
    knowledge_type: Literal[KnowledgeTypeEnum.QA] = Field(
        KnowledgeTypeEnum.QA, description="type of knowledge resource"
    )
    question: str = Field(..., description="question of the knowledge resource")
    answer: str = Field(..., description="answer of the knowledge resource")
    split_config: TextSplitConfig = Field(
        ...,
        description="split config of the knowledge",
    )
    source_config: Optional[TextSourceConfig] = Field(
        default=None,
        description="source config of the knowledge",
    )

    @model_validator(mode="after")
    def update_source_and_metadata(self) -> "QACreate":
        self.source_config = self.source_config or TextSourceConfig()
        self.source_config.text = self.question

        self.metadata = self.metadata or {}
        self.metadata["answer"] = self.answer

        return self


class ImageCreate(KnowledgeCreateBase):
    knowledge_type: Literal[KnowledgeTypeEnum.IMAGE] = Field(
        KnowledgeTypeEnum.IMAGE, description="type of knowledge resource"
    )
    source_config: Union[OpenUrlSourceConfig, OpenIdSourceConfig, S3SourceConfig] = (
        Field(
            ...,
            description="source config of the knowledge",
        )
    )
    file_sha: str = Field(..., description="SHA of the file")
    file_size: int = Field(..., description="Byte size of the file")


# TODO: add more knowledge types
# EXCEL = "excel"
# PPTX = "pptx"
# 定义一个类型变量来表示所有可能的知识创建类型
KnowledgeCreateUnion = Union[
    TextCreate,
    ImageCreate,
    JSONCreate,
    MarkdownCreate,
    PDFCreate,
    GithubRepoCreate,
    QACreate,
]
