import os

import pytest

from whiskerrag_types.model.knowledge import Knowledge
from whiskerrag_utils import RegisterTypeEnum, get_register
from whiskerrag_utils.registry import init_register

knowledge_data = {
    "source_type": "user_input_text",
    "knowledge_type": "text",
    "space_id": "local_test",
    "knowledge_name": "local_test_5",
    "split_config": {"chunk_size": 2000, "chunk_overlap": 100},
    "source_config": {"text": "hello world"},
    "embedding_model_name": "openai",
    "tenant_id": "38fbd78b-1869-482c-9142-e43a2c2s6e42",
    "metadata": {
        "_reference_url": "test",
    },
}


class TestTextLoader:
    @pytest.mark.asyncio
    async def test_get_user_input_loader(self) -> None:
        knowledge = Knowledge(**knowledge_data)
        os.environ["OPENAI_API_KEY"] = "test_openai_api_key"
        init_register("whiskerrag_utils")
        LoaderCls = get_register(
            RegisterTypeEnum.KNOWLEDGE_LOADER, knowledge.source_type
        )
        res = await LoaderCls(knowledge).load()
        assert res[0].content == "hello world"
        assert res[0].metadata["_reference_url"] == "test"
