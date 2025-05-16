import os

import pytest

from whiskerrag_types.model.knowledge import (
    Knowledge,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
    OpenUrlSourceConfig,
    TextSourceConfig,
)
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils import RegisterTypeEnum, get_register
from whiskerrag_utils.registry import init_register


class TestJSONLoader:
    @pytest.mark.skip(reason="Skipping this test temporarily")
    @pytest.mark.asyncio
    async def test_get_cloud_url_json(self) -> None:
        knowledge_data = {
            "source_type": KnowledgeSourceEnum.CLOUD_STORAGE_TEXT,
            "knowledge_type": KnowledgeTypeEnum.JSON,
            "space_id": "local_test",
            "knowledge_name": "yuque",
            "split_config": {
                "type": "json",
                "max_chunk_size": 3000,
                "min_chunk_size": 1000,
            },
            "source_config": OpenUrlSourceConfig(
                url="https://xxx.json",
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
        print("----len", len(res[0].content))
        SplitterCls = get_register(RegisterTypeEnum.SPLITTER, "json")
        res = SplitterCls().parse(knowledge, res[0])
        print("----len", len(res))

    @pytest.mark.asyncio
    async def test_user_input_json(self) -> None:
        json_str = """
        {
            "name": "John DoeJohn DoeJohn DoeJohn DoeJohn DoeJohn DoeJohn Doe",
            "age": 30,
            "email": "johnjohnjohnjohnjohnjohnjohn@example.com",
            "is_active": true
        }
        """
        knowledge_data = {
            "source_type": KnowledgeSourceEnum.USER_INPUT_TEXT,
            "knowledge_type": KnowledgeTypeEnum.JSON,
            "space_id": "local_test",
            "knowledge_name": "json",
            "split_config": {
                "type": "json",
                "max_chunk_size": 10,
                "min_chunk_size": 4,
            },
            "source_config": TextSourceConfig(
                text=json_str,
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
        SplitterCls = get_register(RegisterTypeEnum.SPLITTER, "json")
        res = SplitterCls().parse(knowledge, res[0])
        assert res == [
            Text(
                content='{"name": "John DoeJohn DoeJohn DoeJohn DoeJohn DoeJohn DoeJohn Doe"}',
                metadata={},
            ),
            Text(content='{"age": 30}', metadata={}),
            Text(
                content='{"email": "johnjohnjohnjohnjohnjohnjohn@example.com"}',
                metadata={},
            ),
            Text(content='{"is_active": true}', metadata={}),
        ]
