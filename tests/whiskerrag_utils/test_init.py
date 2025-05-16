from unittest.mock import AsyncMock, patch

import pytest

from whiskerrag_types.model.knowledge import Knowledge
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils import get_chunks_by_knowledge
from whiskerrag_utils.registry import RegisterTypeEnum


class MockLoader:
    def __init__(self, knowledge) -> None:
        self.knowledge = knowledge

    async def load(self):
        return [
            Text(content="content1", metadata={}),
            Text(content="content2", metadata={}),
        ]


class MockSplitter:
    call_count = 0

    def __init__(self) -> None:
        pass

    def parse(self, knowledge, content):
        return [
            Text(content="split1", metadata={"key1": "value11", "key2": "value22"}),
            Text(content="split2", metadata={"key1": "value11", "key2": "value22"}),
        ]


class MockEmbedding:
    def __init__(self):
        pass

    async def embed_text(self, text, timeout=None):
        return [0, 1]


@pytest.mark.asyncio
async def test_get_chunks_by_knowledge_text() -> None:
    data = {
        "knowledge_id": "bb787386-ed19-4cc8-966d-9ed63bb7993c",
        "space_id": "OpenSPG/KAG",
        "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c226e42",
        "knowledge_type": "markdown",
        "knowledge_name": "OpenSPG/KAG/kag/examples/supplychain/builder/README.md",
        "source_type": "github_file",
        "source_config": {
            "repo_name": "OpenSPG/KAG",
            "branch": None,
            "commit_id": None,
            "auth_info": None,
            "path": "kag/examples/supplychain/builder/README.md",
        },
        "embedding_model_name": "openai",
        "split_config": {
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "separators": None,
            "split_regex": None,
        },
        "file_sha": "d592ae35c26e3cbc393080b7549e7754e7206f69",
        "file_size": 3961,
        "metadata": {
            "key1": "value1",
        },
        "retrieval_count": 0,
        "parent_id": "155555f9-f4d9-471e-9fbd-67239fbe371b",
        "enabled": True,
        "created_at": "2025-05-03T02:14:09.347451Z",
        "updated_at": "2025-05-03T02:14:09.347451Z",
    }
    knowledge = Knowledge(**data)
    mock_embedding_instance = AsyncMock()
    mock_embedding_instance.embed_text = AsyncMock(
        side_effect=[["embedding1"], ["embedding2"]]
    )
    with patch(
        "whiskerrag_utils.get_register",
        side_effect=lambda *args: {
            RegisterTypeEnum.KNOWLEDGE_LOADER: MockLoader,
            RegisterTypeEnum.SPLITTER: MockSplitter,
            RegisterTypeEnum.EMBEDDING: MockEmbedding,
        }[args[0]],
    ):
        chunks = await get_chunks_by_knowledge(knowledge)
        assert len(chunks) == 4
        assert chunks[0].context == "split1"
        assert chunks[0].metadata == {"key1": "value11", "key2": "value22"}
