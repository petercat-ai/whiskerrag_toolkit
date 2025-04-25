from whiskerrag_types.model.knowledge import Knowledge, KnowledgeTypeEnum
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
        SplitterCls = get_register(RegisterTypeEnum.SPLITTER, knowledge.knowledge_type)
        res = SplitterCls().split(json_str, knowledge.split_config)
        assert res == [
            '{"name": "John DoeJohn DoeJohn DoeJohn DoeJohn DoeJohn DoeJohn Doe"}',
            '{"age": 30}',
            '{"email": "johnjohnjohnjohnjohnjohnjohn@example.com"}',
            '{"is_active": true}',
        ]
