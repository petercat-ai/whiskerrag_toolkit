import pytest

from whiskerrag_types.model.knowledge import Knowledge
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils import RegisterTypeEnum, get_register
from whiskerrag_utils.registry import init_register


class TestTextSplitter:
    @pytest.mark.asyncio
    async def test_text_split(self) -> None:
        knowledge_data = {
            "source_type": "user_input_text",
            "knowledge_type": "text",
            "space_id": "local_test",
            "knowledge_name": "local_test_5",
            "split_config": {
                "type": "text",
                "chunk_size": 5,
                "chunk_overlap": 0,
                "separators": ["\n\n", "\n", " "],
                "keep_separator": False,
                "is_separator_regex": False,
            },
            "source_config": {"text": "hello world"},
            "embedding_model_name": "openai",
            "tenant_id": "38fbd78b-1869-482c-9142-e43a2c2s6e42",
            "metadata": {},
        }

        knowledge = Knowledge(**knowledge_data)
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.PARSER, "text")
        res = await SplitterCls().parse(
            knowledge, Text(content="hello world \n ~", metadata={})
        )

        assert res == [
            Text(content="hello", metadata={}),
            Text(content="world", metadata={}),
            Text(content="~", metadata={}),
        ]

    @pytest.mark.asyncio
    async def test_text_split_regex(self) -> None:
        knowledge_data = {
            "source_type": "user_input_text",
            "knowledge_type": "text",
            "space_id": "local_test",
            "knowledge_name": "local_test_5",
            "split_config": {
                "type": "text",
                "chunk_size": 5,
                "chunk_overlap": 0,
                "separators": ["\\n\\n", "\\n", "\\s"],
                "is_separator_regex": True,
                "keep_separator": False,
            },
            "source_config": {"text": "hello world"},
            "embedding_model_name": "openai",
            "tenant_id": "38fbd78b-1869-482c-9142-e43a2c2s6e42",
            "metadata": {},
        }

        knowledge = Knowledge(**knowledge_data)
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.PARSER, "text")
        res = await SplitterCls().parse(
            knowledge, Text(content="hello world \n ~", metadata={})
        )

        assert res == [
            Text(content="hello", metadata={}),
            Text(content="world", metadata={}),
            Text(content="~", metadata={}),
        ]
