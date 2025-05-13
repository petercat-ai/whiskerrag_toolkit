import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from whiskerrag_types.model.knowledge import (
    Knowledge,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
    YuqueSourceConfig,
)
from whiskerrag_utils import RegisterTypeEnum, get_register
from whiskerrag_utils.registry import init_register


@pytest.fixture
def mock_yuque_loader() -> Any:
    mock_document = Document(page_content="hello world", metadata={})

    patches = [
        patch("whiskerrag_utils.helper.yuque.ExtendedYuqueLoader"),
        patch(
            "whiskerrag_utils.helper.yuque.ExtendedYuqueLoader.load_document_by_path",
            return_value=mock_document,
        ),
    ]

    with patch.multiple(
        "whiskerrag_utils.helper.yuque.ExtendedYuqueLoader",
        load_document_by_path=MagicMock(return_value=mock_document),
    ) as mocks:
        yield mocks


class TestYuqueLoader:
    @pytest.mark.asyncio
    async def test_get_text(self, mock_yuque_loader: Any) -> None:
        knowledge_data = {
            "source_type": KnowledgeSourceEnum.YUQUE,
            "knowledge_type": KnowledgeTypeEnum.YUQUEDOC,
            "space_id": "local_test",
            "knowledge_name": "yuque",
            "split_config": {"chunk_size": 2000, "chunk_overlap": 100},
            "source_config": YuqueSourceConfig(
                auth_info="xxx",
                api_url="https://yuque-api.xxx.com",
                group_login="xxx",
                book_slug="xxx",
                document_id="xxx",
            ),
            "embedding_model_name": "openai",
            "tenant_id": "38fbd78b-1869-482c-9142-e43a2c2s6e42",
            "metadata": {},
        }
        knowledge = Knowledge(**knowledge_data)
        os.environ["OPENAI_API_KEY"] = "test_openai_api_key"
        init_register("whiskerrag_utils")
        LoaderCls = get_register(
            RegisterTypeEnum.KNOWLEDGE_LOADER, knowledge.source_type
        )
        res = await LoaderCls(knowledge).load()
        assert res[0].content == "hello world"
