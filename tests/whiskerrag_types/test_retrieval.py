import pytest
from pydantic import ValidationError

from whiskerrag_types.model.knowledge import EmbeddingModelEnum
from whiskerrag_types.model.retrieval import RetrievalBySpaceRequest


class TestRetrieval:
    def test_str_to_RetrievalRequest(self) -> None:
        data = {
            "embedding_model_name": "bge-base-chinese-1117",
            "question": "如何申请休假？",
            "space_id_list": ["hr_space", "policy_space"],
            "top": 5,
            "similarity_threshold": 0.9,
            "metadata_filter": {},
        }
        res = RetrievalBySpaceRequest(**data)
        assert res.embedding_model_name == "bge-base-chinese-1117"
        assert res.question == "如何申请休假？"
        assert res.space_id_list == ["hr_space", "policy_space"]
        assert res.top == 5
        assert res.similarity_threshold == 0.9
        assert res.metadata_filter == {}

    def test_none_case(self) -> None:
        data = {
            "embedding_model_name": None,
            "question": "如何申请休假？",
            "space_id_list": ["hr_space", "policy_space"],
            "top": 5,
            "similarity_threshold": 0.9,
            "metadata_filter": {},
        }
        with pytest.raises(ValidationError):
            RetrievalBySpaceRequest(**data)

    def test_enum_to_RetrievalRequest(self) -> None:
        data = {
            "embedding_model_name": EmbeddingModelEnum.OPENAI,
            "question": "如何申请休假？",
            "space_id_list": ["hr_space", "policy_space"],
            "top": 5,
            "similarity_threshold": 0.9,
            "metadata_filter": {},
        }
        res = RetrievalBySpaceRequest(**data)
        assert res.embedding_model_name == "openai"
        assert res.question == "如何申请休假？"
        assert res.space_id_list == ["hr_space", "policy_space"]
        assert res.top == 5
        assert res.similarity_threshold == 0.9
        assert res.metadata_filter == {}
