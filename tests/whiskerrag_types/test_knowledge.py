from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
    Knowledge,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
    TextSourceConfig,
)
from whiskerrag_types.model.utils import calculate_sha256


class TestKnowledge:
    def test_knowledge_initialization_with_sha256(self) -> None:
        text = "This is a test text for knowledge creation."
        data = {
            "space_id": "test_space",
            "knowledge_type": KnowledgeTypeEnum.TEXT,
            "knowledge_name": "Test Knowledge",
            "source_type": KnowledgeSourceEnum.USER_INPUT_TEXT,
            "source_config": TextSourceConfig(text=text),
            "embedding_model_name": EmbeddingModelEnum.OPENAI,
            "split_config": {
                "type": "text",
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": [],
                "split_regex": None,
                "strip_whitespace": True,
            },
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
        }
        knowledge = Knowledge(**data)
        assert knowledge.file_sha == calculate_sha256(text)
        assert knowledge.file_size == len(text.encode("utf-8"))
        assert knowledge.split_config.type == "text"

    def test_knowledge_update(self) -> None:
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
                "separators": None,
                "split_regex": None,
                "strip_whitespace": True,
            },
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
        }
        knowledge = Knowledge(**data)
        updated_data = {"knowledge_name": "Updated Knowledge"}
        knowledge.update(**updated_data)
        assert knowledge.knowledge_name == "Updated Knowledge"

    def test_knowledge_init_by_old_version(self) -> None:
        text = "This is a test text for knowledge creation."
        data = {
            "space_id": "test_space",
            "knowledge_type": KnowledgeTypeEnum.TEXT,
            "knowledge_name": "Test Knowledge",
            "source_type": KnowledgeSourceEnum.USER_INPUT_TEXT,
            "source_config": TextSourceConfig(text=text),
            "embedding_model_name": EmbeddingModelEnum.OPENAI,
            "split_config": {
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": None,
                "split_regex": None,
            },
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
        }
        knowledge = Knowledge(**data)
        assert knowledge.split_config.chunk_size == 500
        assert knowledge.split_config.chunk_overlap == 100
        assert knowledge.split_config.separators is None
        assert knowledge.split_config.split_regex is None
        assert knowledge.split_config.model_dump() == {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "separators": None,
            "split_regex": None,
        }

    def test_knowledge_model_dump(self) -> None:
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
                "separators": None,
                "split_regex": None,
                "strip_whitespace": True,
            },
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
            "gmt_create": "2023-01-01T00:00:00Z",
        }
        knowledge = Knowledge(**data).model_dump()
        # time use created_at instead of gmt_create
        assert knowledge["created_at"] == "2023-01-01T00:00:00.000000Z"
        assert knowledge["split_config"] == {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "separators": None,
            "split_regex": None,
            "type": "text",
            "keep_separator": False,
            "strip_whitespace": True,
        }
        assert knowledge["updated_at"] is not None

    def test_init_knowledge(self) -> None:
        data = {
            "knowledge_id": "425c7431-bade-4eac-8e81-1d6dcd8113b7",
            "space_id": "e16c63c3-5899-4c4d-9a09-c50c1d7cd7d2",
            "tenant_id": "db224d11-0d15-4afb-bebe-30a670edbe13",
            "knowledge_type": "text",
            "knowledge_name": "eyre7222",
            "parent_id": None,
            "source_type": "user_input_text",
            "source_config": {"text": "this is test aaaaa\n"},
            "embedding_model_name": "bge-base-chinese-1117",
            "split_config": {
                "chunk_size": 1024,
                "chunk_overlap": 1000,
                "separators": ["\n\n"],
                "split_regex": None,
                "type": "text",
                "keep_separator": True,
                "strip_whitespace": True,
            },
            "file_sha": "b25d83386bb4ff85c97dede5e831b0de2d6a38326c52f2abab3b2fe88eb1c2d5",
            "file_size": 19,
            "metadata": {},
            "enabled": 1,
            "created_at": "2025-04-28T03:03:48.000000Z",
            "updated_at": "2025-04-28T03:03:48.000000Z",
            "retrieval_count": 0,
        }
        Knowledge(**data)
