from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
)
from whiskerrag_types.model.knowledge_create import ImageCreate, QACreate, TextCreate


class TestKnowledge:
    def test_TextCreate(self) -> None:
        data = {
            "space_id": "test_space",
            "knowledge_type": KnowledgeTypeEnum.TEXT,
            "knowledge_name": "Test Knowledge",
            "source_type": KnowledgeSourceEnum.USER_INPUT_TEXT,
            "source_config": {"text": "This is a test text for knowledge creation."},
            "embedding_model_name": EmbeddingModelEnum.OPENAI,
            "split_config": {
                "type": "text",
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": ["\n\n", "##"],
                "strip_whitespace": True,
                "keep_separator": False,
            },
        }
        knowledge_create = TextCreate(**data)
        assert knowledge_create.space_id == "test_space"
        assert knowledge_create.knowledge_type == KnowledgeTypeEnum.TEXT
        assert knowledge_create.knowledge_name == "Test Knowledge"
        assert knowledge_create.source_type == KnowledgeSourceEnum.USER_INPUT_TEXT
        assert (
            knowledge_create.source_config.text
            == "This is a test text for knowledge creation."
        )
        assert knowledge_create.embedding_model_name == EmbeddingModelEnum.OPENAI
        assert knowledge_create.split_config.type == "text"
        assert knowledge_create.split_config.chunk_size == 500
        assert knowledge_create.split_config.chunk_overlap == 100
        assert knowledge_create.split_config.separators == ["\n\n", "##"]
        assert knowledge_create.split_config.strip_whitespace is True
        assert knowledge_create.split_config.keep_separator is False

    def test_QACreate(self) -> None:
        text = "This is a test text for knowledge creation."
        data = {
            "knowledge_name": "test_qa",
            "space_id": "test_space",
            "source_type": KnowledgeSourceEnum.USER_INPUT_TEXT,
            "knowledge_type": KnowledgeTypeEnum.QA,
            "question": text,
            "answer": "world",
            "split_config": {
                "type": "text",
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": ["\n\n", "##"],
                "strip_whitespace": True,
                "keep_separator": False,
            },
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
        }
        knowledge = QACreate(**data)
        assert knowledge.source_config.text == text
        assert knowledge.metadata == {"answer": "world"}

    def test_ImageCreate(self) -> None:
        data = {
            "knowledge_name": "test_qa",
            "space_id": "test_space",
            "source_type": KnowledgeSourceEnum.CLOUD_STORAGE_IMAGE,
            "knowledge_type": KnowledgeTypeEnum.IMAGE,
            "source_config": {"url": "112233"},
            "split_config": {"type": "image"},
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
            "file_size": 122,
            "file_sha": "123213",
        }
        knowledge = ImageCreate(**data)
        assert knowledge.knowledge_type == KnowledgeTypeEnum.IMAGE
