from unittest.mock import patch

import pytest

from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
    Knowledge,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
    YuqueSourceConfig,
)
from whiskerrag_utils.decomposer.folder_decomposer import FolderDecomposer


@pytest.fixture
def mock_yuque_loader():
    with patch(
        "whiskerrag_utils.decomposer.folder_decomposer.YuqueLoader"
    ) as mock_loader:
        mock_document = {
            "id": 1234,
            "title": "Test Document",
            "content": "Test Content",
        }

        mock_books = [
            {
                "id": 5678,
                "title": "Test Book 1",
                "items": [mock_document],
            },
            {
                "id": 5679,
                "title": "Test Book 2",
                "items": [mock_document],
            },
        ]

        mock_documents = [
            {
                "id": 1234,
                "title": "Test Document 1",
            },
            {
                "id": 1235,
                "title": "Test Document 2",
            },
        ]

        # set mock function
        instance = mock_loader.return_value
        instance.get_documents = mock_documents
        instance.get_document.return_value = mock_document
        instance.get_books.return_value = mock_books
        instance.get_document_ids.return_value = [1234, 1235]
        instance.get_user_id.return_value = "test_user"

        yield mock_loader


@pytest.fixture
def base_knowledge() -> Knowledge:
    data = {
        "space_id": "test_space",
        "knowledge_type": KnowledgeTypeEnum.FOLDER,
        "knowledge_name": "TestYuque",
        "source_type": KnowledgeSourceEnum.YUQUE,
        "source_config": YuqueSourceConfig(
            api_url="https://www.yuque.com",
            group_id=12345,
            auth_info="test_token",
        ),
        "embedding_model_name": EmbeddingModelEnum.OPENAI,
        "split_config": {
            "type": "yuque",
            "chunk_size": 500,
            "chunk_overlap": 100,
            "separators": [],
            "split_regex": None,
        },
        "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
    }
    return Knowledge(**data)


class TestGetKnowledgeListFromYuque:
    @pytest.mark.asyncio
    async def test_document_level(self, mock_yuque_loader, base_knowledge) -> None:
        base_knowledge.source_config.book_id = 5678
        base_knowledge.source_config.document_id = 1234

        instance = FolderDecomposer(base_knowledge)

        result = await instance.decompose()

        assert len(result) == 1
        assert result[0].knowledge_type == KnowledgeTypeEnum.YUQUEDOC
        assert result[0].source_type == KnowledgeSourceEnum.YUQUE
        assert result[0].knowledge_name == "Test Document"
        assert result[0].source_config.document_id == 1234

    @pytest.mark.asyncio
    async def test_book_level(self, mock_yuque_loader, base_knowledge):
        base_knowledge.source_config.book_id = 5678

        instance = FolderDecomposer(base_knowledge)

        result = await instance.decompose()

        assert len(result) == 2
        for knowledge in result:
            assert knowledge.knowledge_type == KnowledgeTypeEnum.YUQUEDOC
            assert knowledge.source_type == KnowledgeSourceEnum.YUQUE
            assert knowledge.source_config.book_id == 5678

    @pytest.mark.asyncio
    async def test_group_level(self, mock_yuque_loader, base_knowledge) -> None:
        base_knowledge.source_config.document_id = None
        base_knowledge.source_config.book_id = None
        instance = FolderDecomposer(base_knowledge)

        result = await instance.decompose()

        assert len(result) > 0
        for knowledge in result:
            assert knowledge.knowledge_type == KnowledgeTypeEnum.YUQUEDOC
            assert knowledge.source_type == KnowledgeSourceEnum.YUQUE
            assert knowledge.source_config.group_id == 12345

    @pytest.mark.asyncio
    async def test_document_without_book(
        self, mock_yuque_loader, base_knowledge
    ) -> None:
        base_knowledge.source_config.document_id = 1234
        base_knowledge.source_config.book_id = None

        instance = FolderDecomposer(base_knowledge)

        with pytest.raises(Exception):
            await instance.get_knowledge_list_from_yuque(base_knowledge)

    @pytest.mark.asyncio
    async def test_yuque_api_error(self, mock_yuque_loader, base_knowledge) -> None:
        mock_yuque_loader.return_value.get_document.side_effect = Exception("API Error")

        base_knowledge.source_config.book_id = 5678
        base_knowledge.source_config.document_id = 1234

        instance = FolderDecomposer(base_knowledge)

        with pytest.raises(Exception) as exc_info:
            await instance.get_knowledge_list_from_yuque(base_knowledge)
        assert "Failed to get knowledge list from Yuque" in str(exc_info.value)
