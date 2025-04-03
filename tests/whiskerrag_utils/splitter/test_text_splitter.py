from whiskerrag_types.model.knowledge import Knowledge
from whiskerrag_utils import RegisterTypeEnum, get_register
from whiskerrag_utils.registry import init_register

knowledge_data = {
    "source_type": "user_input_text",
    "knowledge_type": "text",
    "space_id": "local_test",
    "knowledge_name": "local_test_5",
    "split_config": {
        "chunk_size": 1,
        "chunk_overlap": 0,
        "separators": ["\n\n", "\n", " "],
    },
    "source_config": {"text": "hello world"},
    "embedding_model_name": "openai",
    "tenant_id": "38fbd78b-1869-482c-9142-e43a2c2s6e42",
    "metadata": {},
}


# pytest tests/whiskerrag_utils/splitter/test_text_splitter.py
class TestTextSplitter:
    def test_text_split(self) -> None:
        knowledge = Knowledge(**knowledge_data)
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.SPLITTER, knowledge.knowledge_type)
        res = SplitterCls().split("hello world \n ~", knowledge.split_config)
        print("---res", res)
        assert res == ["hello", "world", "~"]
