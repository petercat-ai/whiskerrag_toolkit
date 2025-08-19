import pytest

from whiskerrag_types.model.knowledge import Knowledge
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils import RegisterTypeEnum, get_register
from whiskerrag_utils.registry import init_register


class TestBaseSplitter:

    @pytest.mark.asyncio
    async def test_base_split(self) -> None:
        knowledge_data = {
            "source_type": "user_input_text",
            "knowledge_type": "text",
            "space_id": "local_test",
            "knowledge_name": "local_test_5",
            "split_config": {
                "chunk_size": 100,
                "chunk_overlap": 0,
            },
            "source_config": {"text": "hello world"},
            "embedding_model_name": "openai",
            "tenant_id": "38fbd78b-1869-482c-9142-e43a2c2s6e42",
            "metadata": {},
        }

        knowledge = Knowledge(**knowledge_data)
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.PARSER, "base_text")
        res = await SplitterCls().parse(
            knowledge, Text(content="hello world \n ~", metadata={})
        )
        expected_metadata = {
            "_knowledge_type": "text",
            "_reference_url": "",
            "_knowledge_name": "local_test_5",
            "_idx": 0,
        }
        assert res == [Text(content="hello world \n ~", metadata=expected_metadata)]

    @pytest.mark.asyncio
    async def test_base_split_small_chunk(self) -> None:
        knowledge_data = {
            "source_type": "user_input_text",
            "knowledge_type": "text",
            "space_id": "local_test",
            "knowledge_name": "local_test_5",
            "split_config": {
                "chunk_size": 5,
                "chunk_overlap": 1,
            },
            "source_config": {"text": "hello world"},
            "embedding_model_name": "openai",
            "tenant_id": "38fbd78b-1869-482c-9142-e43a2c2s6e42",
            "metadata": {},
        }

        knowledge = Knowledge(**knowledge_data)
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.PARSER, "base_text")
        res = await SplitterCls().parse(
            knowledge, Text(content="hello world \n ~", metadata={})
        )
        assert res == [
            Text(
                content="hello",
                metadata={
                    "_knowledge_type": "text",
                    "_knowledge_name": "local_test_5",
                    "_reference_url": "",
                    "_idx": 0,
                },
            ),
            Text(
                content="world",
                metadata={
                    "_knowledge_type": "text",
                    "_knowledge_name": "local_test_5",
                    "_reference_url": "",
                    "_idx": 1,
                },
            ),
            Text(
                content="~",
                metadata={
                    "_knowledge_type": "text",
                    "_knowledge_name": "local_test_5",
                    "_reference_url": "",
                    "_idx": 2,
                },
            ),
        ]

    @pytest.mark.asyncio
    async def test_base_split_2(self) -> None:
        knowledge_data = {
            "source_type": "user_input_text",
            "knowledge_type": "text",
            "space_id": "local_test",
            "knowledge_name": "local_test_5",
            "split_config": {
                "chunk_size": 1,
                "chunk_overlap": 0,
                "separators": [
                    "\n#{1,6} ",
                    "```\n",
                    "\n\\*\\*\\*+\n",
                    "\n---+\n",
                    "\n___+\n",
                    "\n\n",
                    "\n",
                    " ",
                ],
                "split_regex": None,
            },
            "source_config": {"text": "hello world"},
            "embedding_model_name": "openai",
            "tenant_id": "38fbd78b-1869-482c-9142-e43a2c2s6e42",
            "metadata": {},
        }

        knowledge = Knowledge(**knowledge_data)
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.PARSER, "base_text")
        res = await SplitterCls().parse(
            knowledge,
            Text(content="hello world \n ~", metadata={}),
        )
        assert res[0] == Text(
            content="h",
            metadata={
                "_knowledge_type": "text",
                "_reference_url": "",
                "_knowledge_name": "local_test_5",
                "_idx": 0,
            },
        )
