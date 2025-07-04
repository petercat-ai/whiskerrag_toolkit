import pytest

from whiskerrag_types.model.knowledge import Knowledge, KnowledgeTypeEnum
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils.registry import RegisterTypeEnum, get_register, init_register

json_str = """
{
    "name": "John DoeJohn DoeJohn DoeJohn DoeJohn DoeJohn DoeJohn Doe",
    "age": 30,
    "email": "johnjohnjohnjohnjohnjohnjohn@example.com",
    "is_active": true
}
"""

# Test array format
json_array_str = """
[
    {"id": 1, "name": "Item 1"},
    {"id": 2, "name": "Item 2"},
    {"id": 3, "name": "Item 3"}
]
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

    @pytest.mark.asyncio
    async def test_json_split(self) -> None:
        knowledge = Knowledge(**knowledge_data)
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.PARSER, "json")
        res = await SplitterCls().parse(
            knowledge,
            Text(
                content=json_str,
                metadata=knowledge.metadata,
            ),
        )
        assert res == [
            Text(
                content='{"name": "John DoeJohn DoeJohn DoeJohn DoeJohn DoeJohn DoeJohn Doe"}',
                metadata={"_knowledge_type": "json", "_reference_url": ""},
            ),
            Text(
                content='{"age": 30}',
                metadata={"_knowledge_type": "json", "_reference_url": ""},
            ),
            Text(
                content='{"email": "johnjohnjohnjohnjohnjohnjohn@example.com"}',
                metadata={"_knowledge_type": "json", "_reference_url": ""},
            ),
            Text(
                content='{"is_active": true}',
                metadata={"_knowledge_type": "json", "_reference_url": ""},
            ),
        ]

    @pytest.mark.asyncio
    async def test_json_array_split(self) -> None:
        """Test JSON array format support"""
        knowledge = Knowledge(**knowledge_data)
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.PARSER, "json")
        res = await SplitterCls().parse(
            knowledge,
            Text(
                content=json_array_str,
                metadata=knowledge.metadata,
            ),
        )
        # Array should be converted to dict with "data" key
        assert len(res) > 0
        # Verify that the result contains the array data
        assert any("Item 1" in text.content for text in res)
        assert any("Item 2" in text.content for text in res)
        assert any("Item 3" in text.content for text in res)

    @pytest.mark.asyncio
    async def test_json_split_error(self) -> None:
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.PARSER, "json")
        knowledge = Knowledge(**knowledge_data)
        with pytest.raises(ValueError) as excinfo:
            await SplitterCls().parse(
                knowledge,
                Text(content="4", metadata={}),
            )
        assert (
            str(excinfo.value)
            == "Error processing JSON content: JSON content must be a dictionary or array."
        )

    @pytest.mark.asyncio
    async def test_invalid_json_error(self) -> None:
        """Test invalid JSON format error handling"""
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.PARSER, "json")
        knowledge = Knowledge(**knowledge_data)
        with pytest.raises(ValueError) as excinfo:
            await SplitterCls().parse(
                knowledge,
                Text(content='{"invalid": json}', metadata={}),
            )
        assert "Invalid JSON content provided for splitting" in str(excinfo.value)
