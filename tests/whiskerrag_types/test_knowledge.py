from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
    Knowledge,
    KnowledgeCreate,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
    TextSourceConfig,
    calculate_sha256,
)


class TestKnowledge:
    def test_knowledge_create_initialization(self) -> None:
        data = {
            "space_id": "test_space",
            "knowledge_type": KnowledgeTypeEnum.TEXT,
            "knowledge_name": "Test Knowledge",
            "source_type": KnowledgeSourceEnum.USER_INPUT_TEXT,
            "source_config": {"text": "This is a test text for knowledge creation."},
            "embedding_model_name": EmbeddingModelEnum.OPENAI,
            "split_config": {
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": None,
                "split_regex": None,
                "strip_whitespace": True,
            },
        }
        knowledge_create = KnowledgeCreate(**data)
        assert knowledge_create.space_id == "test_space"
        assert knowledge_create.knowledge_type == KnowledgeTypeEnum.TEXT
        assert knowledge_create.knowledge_name == "Test Knowledge"
        assert knowledge_create.source_type == KnowledgeSourceEnum.USER_INPUT_TEXT
        assert (
            knowledge_create.source_config.text
            == "This is a test text for knowledge creation."
        )
        assert knowledge_create.embedding_model_name == EmbeddingModelEnum.OPENAI
        assert knowledge_create.split_config.chunk_size == 500

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
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": None,
                "split_regex": None,
                "strip_whitespace": True,
            },
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
        }
        knowledge = Knowledge(**data)
        assert knowledge.file_sha == calculate_sha256(text)
        assert knowledge.file_size == len(text.encode("utf-8"))

    def test_knowledge_update(self) -> None:
        data = {
            "space_id": "test_space",
            "knowledge_type": KnowledgeTypeEnum.TEXT,
            "knowledge_name": "Test Knowledge",
            "source_type": KnowledgeSourceEnum.USER_INPUT_TEXT,
            "source_config": TextSourceConfig(text="Initial text."),
            "embedding_model_name": EmbeddingModelEnum.OPENAI,
            "split_config": {
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
