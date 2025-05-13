from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
    Knowledge,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
    TextSourceConfig,
)


class TestKnowledge:
    def test_default_char_split(self) -> None:
        data = {
            "space_id": "test_space",
            "knowledge_type": KnowledgeTypeEnum.YUQUEDOC,
            "knowledge_name": "Test Knowledge",
            "source_type": KnowledgeSourceEnum.USER_INPUT_TEXT,
            "source_config": TextSourceConfig(text="Initial text."),
            "embedding_model_name": EmbeddingModelEnum.OPENAI,
            "split_config": {
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": None,
                "split_regex": None,
            },
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
            "gmt_create": "2023-01-01T00:00:00Z",
            "gmt_modified": "2023-01-01T00:00:00Z",
        }
        knowledge = Knowledge(**data).model_dump()
        assert knowledge["created_at"] == "2023-01-01T00:00:00.000000Z"
        assert knowledge["split_config"] == {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "separators": None,
            "split_regex": None,
        }
        assert knowledge["updated_at"] is not None

    def test_text_split(self) -> None:
        data = {
            "space_id": "test_space",
            "knowledge_type": KnowledgeTypeEnum.TEXT,
            "knowledge_name": "Test Knowledge",
            "source_type": KnowledgeSourceEnum.USER_INPUT_TEXT,
            "source_config": TextSourceConfig(text="Initial text."),
            "embedding_model_name": EmbeddingModelEnum.OPENAI,
            "split_config": {
                "type": "text",
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": ["\n", "\n\n", "\r"],
                "is_separator_regex": False,
            },
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
            "gmt_create": "2023-01-01T00:00:00Z",
            "gmt_modified": "2023-01-01T00:00:00Z",
        }
        knowledge = Knowledge(**data).model_dump()
        assert knowledge["created_at"] == "2023-01-01T00:00:00.000000Z"
        assert knowledge["updated_at"] is not None

    def test_markdown_split(self) -> None:
        data = {
            "space_id": "test_space",
            "knowledge_type": KnowledgeTypeEnum.MARKDOWN,
            "knowledge_name": "Test Knowledge",
            "source_type": KnowledgeSourceEnum.USER_INPUT_TEXT,
            "source_config": TextSourceConfig(text="Initial text."),
            "embedding_model_name": EmbeddingModelEnum.OPENAI,
            "split_config": {
                "type": "markdown",
                "chunk_size": 200,
                "chunk_overlap": 20,
                "separators": ["\n", "\n\n", "\r"],
                "is_separator_regex": False,
            },
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
            "gmt_create": "2023-01-01T00:00:00Z",
            "gmt_modified": "2023-01-01T00:00:00Z",
        }
        knowledge = Knowledge(**data).model_dump()
        assert knowledge["split_config"]["type"] == "markdown"

    def test_markdown_split_default(self) -> None:
        data = {
            "space_id": "test_space",
            "knowledge_type": KnowledgeTypeEnum.TEXT,
            "knowledge_name": "Test Knowledge",
            "source_type": KnowledgeSourceEnum.USER_INPUT_TEXT,
            "source_config": TextSourceConfig(text="Initial text."),
            "embedding_model_name": EmbeddingModelEnum.OPENAI,
            "split_config": {
                "type": "markdown",
                "chunk_size": 200,
                "chunk_overlap": 20,
                "separators": ["\n", "\n\n", "\r"],
                "is_separator_regex": False,
            },
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
            "gmt_create": "2023-01-01T00:00:00Z",
            "gmt_modified": "2023-01-01T00:00:00Z",
        }
        knowledge = Knowledge(**data).model_dump()
        assert knowledge["split_config"]["type"] == "markdown"
        assert knowledge["split_config"]["chunk_size"] == 200
        assert knowledge["split_config"]["chunk_overlap"] == 20
