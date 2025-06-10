from whiskerrag_types.model.chunk import Chunk
from whiskerrag_types.model.utils import parse_datetime

mock_embedding = "[-0.009458053,0.007847417,-0.0023432274,0.00040088524,-0.0023077508]"
data = {
    "chunk_id": "9464425d-f18c-4553-8450-3a215d54117e",
    "embedding": mock_embedding,
    "context": "---\ntitle: treemap\norder: 0\n---",
    "knowledge_id": "d03f9dbf-ee06-40ee-beba-96851c9c4d59",
    "space_id": "antvis/F2",
    "embedding_model_name": "openai",
    "updated_at": "2025-03-04T14:45:59.191407+00:00",
    "created_at": "2025-03-04T14:46:01.76745+00:00",
    "metadata": {},
    "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c226e42",
}

alias_data = {
    "chunk_id": "9464425d-f18c-4553-8450-3a215d54117e",
    "embedding": mock_embedding,
    "context": "---\ntitle: treemap\norder: 0\n---",
    "knowledge_id": "d03f9dbf-ee06-40ee-beba-96851c9c4d59",
    "space_id": "antvis/F2",
    "embedding_model_name": "openai",
    "gmt_modified": "2025-03-04T14:45:59.191407+00:00",
    "gmt_create": "2025-03-04T14:45:59.191407+00:00",
    "metadata": {},
    "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c226e42",
}


class TestChunk:
    def test_json_to_chunk(self) -> None:
        chunk = Chunk(**data)
        assert chunk.knowledge_id == "d03f9dbf-ee06-40ee-beba-96851c9c4d59"
        assert chunk.space_id == "antvis/F2"
        assert chunk.created_at == parse_datetime("2025-03-04T14:46:01.76745+00:00")
        assert chunk.updated_at == parse_datetime("2025-03-04T14:45:59.191407+00:00")

    def test_alias_json_to_chunk(self) -> None:
        chunk = Chunk(**alias_data)
        assert chunk.knowledge_id == "d03f9dbf-ee06-40ee-beba-96851c9c4d59"
        assert chunk.space_id == "antvis/F2"
        assert chunk.created_at == parse_datetime("2025-03-04T14:45:59.191407+00:00")
        assert chunk.updated_at == parse_datetime("2025-03-04T14:45:59.191407+00:00")

    def test_chunk_with_tags_and_fields(self) -> None:
        """Test chunk creation with tags and f1-f5 fields"""
        data_with_fields = {
            **data,
            "tags": ["tag1", "tag2", "test"],
            "f1": "field1_value",
            "f2": "field2_value",
            "f3": "field3_value",
            "f4": "field4_value",
            "f5": "field5_value",
        }

        chunk = Chunk(**data_with_fields)
        assert chunk.tags == ["tag1", "tag2", "test"]
        assert chunk.f1 == "field1_value"
        assert chunk.f2 == "field2_value"
        assert chunk.f3 == "field3_value"
        assert chunk.f4 == "field4_value"
        assert chunk.f5 == "field5_value"

    def test_chunk_optional_fields_default_none(self) -> None:
        """Test that tags and f1-f5 fields default to None when not provided"""
        chunk = Chunk(**data)
        assert chunk.tags is None
        assert chunk.f1 is None
        assert chunk.f2 is None
        assert chunk.f3 is None
        assert chunk.f4 is None
        assert chunk.f5 is None

    def test_chunk_partial_fields(self) -> None:
        """Test chunk with only some of the optional fields provided"""
        data_partial = {
            **data,
            "tags": ["important"],
            "f1": "only_f1_provided",
            # f2-f5 not provided, should be None
        }

        chunk = Chunk(**data_partial)
        assert chunk.tags == ["important"]
        assert chunk.f1 == "only_f1_provided"
        assert chunk.f2 is None
        assert chunk.f3 is None
        assert chunk.f4 is None
        assert chunk.f5 is None

    def test_chunk_empty_tags_list(self) -> None:
        """Test chunk with empty tags list"""
        data_empty_tags = {
            **data,
            "tags": [],
        }

        chunk = Chunk(**data_empty_tags)
        assert chunk.tags == []

    def test_chunk_update_method(self) -> None:
        """Test chunk update method preserves new fields"""
        chunk = Chunk(**data)

        # Update with new values including our new fields
        updated_chunk = chunk.update(
            context="updated context", tags=["updated", "tags"], f1="updated_f1"
        )

        assert updated_chunk.context == "updated context"
        assert updated_chunk.tags == ["updated", "tags"]
        assert updated_chunk.f1 == "updated_f1"
        # Other f fields should remain None
        assert updated_chunk.f2 is None
        assert updated_chunk.f3 is None
        assert updated_chunk.f4 is None
        assert updated_chunk.f5 is None

    def test_chunk_serialization_with_new_fields(self) -> None:
        """Test that chunk can be serialized and deserialized with new fields"""
        data_with_all_fields = {
            **data,
            "tags": ["serialize", "test"],
            "f1": "serialize_f1",
            "f2": "serialize_f2",
            "f3": "serialize_f3",
            "f4": "serialize_f4",
            "f5": "serialize_f5",
        }

        # Create chunk
        original_chunk = Chunk(**data_with_all_fields)

        # Serialize to dict
        chunk_dict = original_chunk.model_dump()

        # Verify new fields are in serialized data
        assert chunk_dict["tags"] == ["serialize", "test"]
        assert chunk_dict["f1"] == "serialize_f1"
        assert chunk_dict["f2"] == "serialize_f2"
        assert chunk_dict["f3"] == "serialize_f3"
        assert chunk_dict["f4"] == "serialize_f4"
        assert chunk_dict["f5"] == "serialize_f5"

        # Deserialize back to chunk
        recreated_chunk = Chunk(**chunk_dict)

        # Verify all fields match
        assert recreated_chunk.tags == original_chunk.tags
        assert recreated_chunk.f1 == original_chunk.f1
        assert recreated_chunk.f2 == original_chunk.f2
        assert recreated_chunk.f3 == original_chunk.f3
        assert recreated_chunk.f4 == original_chunk.f4
        assert recreated_chunk.f5 == original_chunk.f5
