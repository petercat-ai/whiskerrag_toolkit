import pytest

from whiskerrag_types.model.knowledge import Knowledge, KnowledgeTypeEnum
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_types.model.splitter import JSONSplitConfig
from whiskerrag_utils.registry import RegisterTypeEnum, get_register, init_register

json_str = """
{
    "name": "John DoeJohn DoeJohn DoeJohn DoeJohn DoeJohn DoeJohn Doe",
    "age": 30,
    "email": "johnjohnjohnjohnjohnjohnjohn@example.com",
    "is_active": true
}
"""

knowledge_data = {
    "source_type": "user_input_text",
    "knowledge_type": KnowledgeTypeEnum.JSON,
    "space_id": "local_test",
    "knowledge_name": "local_test_5",
    "split_config": {
        "type": "json",
        "max_chunk_size": 10,
        "min_chunk_size": 5,
    },
    "source_config": {"text": json_str},
    "embedding_model_name": "openai",
    "tenant_id": "38fbd78b-1869-482c-9142-e43a2c2s6e42",
    "metadata": {},
}


class TestJSONSplitter:
    def test_json_split(self) -> None:
        knowledge = Knowledge(**knowledge_data)
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.SPLITTER, "json")
        res = SplitterCls().split(
            Text(
                content=json_str,
                metadata=knowledge.metadata,
            ),
            knowledge.split_config,
        )
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

    @pytest.mark.asyncio
    async def test_json_split_error(self) -> None:
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.SPLITTER, "json")
        with pytest.raises(ValueError) as excinfo:
            SplitterCls().split(
                Text(
                    content="4",
                    metadata={},
                ),
                JSONSplitConfig(
                    type="json",
                    max_chunk_size=10,
                    min_chunk_size=5,
                ),
            )
        assert (
            str(excinfo.value)
            == "Error processing JSON content: JSON content must be a dictionary."
        )
