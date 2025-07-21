from typing import List, Optional

import pytest

from whiskerrag_types.interface.embed_interface import BaseEmbedding
from whiskerrag_types.model.knowledge import Knowledge
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils import RegisterTypeEnum, get_register
from whiskerrag_utils.registry import init_register, register


# Mock embedding for testing
class MockEmbedding(BaseEmbedding):
    @classmethod
    async def health_check(cls) -> bool:
        return True  # Always pass health check for testing

    async def embed_documents(
        self, documents: List[str], timeout: Optional[int]
    ) -> List[List[float]]:
        return [[0.1, 0.2, 0.3, 0.4, 0.5]]

    async def embed_text(self, text: str, timeout: Optional[int]) -> List[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    async def embed_text_query(self, text: str, timeout: int = 30) -> List[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    async def embed_image(self, image, timeout: int = 30) -> List[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.5]


class TestMetadataInheritance:

    @pytest.mark.asyncio
    async def test_metadata_inheritance_in_parser(self) -> None:
        """Test that Text.metadata inherits from knowledge.metadata and adds processing info"""
        # Register mock embedding first
        register(RegisterTypeEnum.EMBEDDING, "example")(MockEmbedding)

        # First create knowledge with metadata to see what actually gets stored
        knowledge = Knowledge(
            source_type="user_input_text",
            knowledge_type="text",
            space_id="local_test",
            knowledge_name="local_test_metadata",
            split_config={
                "chunk_size": 100,
                "chunk_overlap": 0,
            },
            source_config={"text": "hello world"},
            embedding_model_name="openai",
            tenant_id="38fbd78b-1869-482c-9142-e43a2c2s6e42",
            metadata={
                "_reference_url": "https://example.com",
                "_tags": "test,parser",
                "_f1": "field1_value",
                "_f2": "field2_value",
                "_f3": "field3_value",
                "_f4": "field4_value",
                "_f5": "field5_value",
                "custom_field": "custom_value",
            },
        )

        init_register()
        ParserCls = get_register(RegisterTypeEnum.PARSER, "base_text")

        # Create Text with some metadata from loader stage
        content = Text(
            content="hello world test content",
            metadata={"loader_info": "from_user_input", "file_path": "/test/path"},
        )

        result = await ParserCls().parse(knowledge, content)

        # Verify that result inherits metadata correctly
        assert len(result) > 0
        chunk = result[0]

        # Should inherit knowledge.metadata - check what's actually there
        assert (
            "_reference_url" in chunk.metadata
        )  # Should be present (even if empty string)
        assert "_knowledge_type" in chunk.metadata
        assert chunk.metadata["_tags"] == "test,parser"
        assert chunk.metadata["_f1"] == "field1_value"
        assert chunk.metadata["_f2"] == "field2_value"
        assert chunk.metadata["_f3"] == "field3_value"
        assert chunk.metadata["_f4"] == "field4_value"
        assert chunk.metadata["_f5"] == "field5_value"
        assert chunk.metadata["custom_field"] == "custom_value"

        # Should inherit content.metadata (from loader stage)
        assert chunk.metadata["loader_info"] == "from_user_input"
        assert chunk.metadata["file_path"] == "/test/path"

    @pytest.mark.asyncio
    async def test_chunk_specific_fields_assignment(self) -> None:
        """Test that chunk gets specific fields from knowledge.metadata._tags, _f1, etc."""
        from whiskerrag_utils import get_chunks_by_knowledge

        # Register mock embedding first
        register(RegisterTypeEnum.EMBEDDING, "example")(MockEmbedding)

        knowledge = Knowledge(
            source_type="user_input_text",
            knowledge_type="text",
            space_id="local_test",
            knowledge_name="local_test_chunk_fields",
            split_config={
                "chunk_size": 50,
                "chunk_overlap": 0,
            },
            source_config={"text": "hello world test content for chunk creation"},
            embedding_model_name="example",  # Use mock embedding
            tenant_id="38fbd78b-1869-482c-9142-e43a2c2s6e42",
            metadata={
                "_reference_url": "https://example.com",
                "_tags": "test,chunk,fields",
                "_f1": "field1_value",
                "_f2": "field2_value",
                "_f3": "field3_value",
                "_f4": "field4_value",
                "_f5": "field5_value",
            },
        )

        init_register()

        # This should work end-to-end
        chunks = await get_chunks_by_knowledge(knowledge)

        assert len(chunks) > 0
        chunk = chunks[0]

        # Verify specific fields are assigned from metadata
        assert chunk.tags == ["test", "chunk", "fields"]
        assert chunk.f1 == "field1_value"
        assert chunk.f2 == "field2_value"
        assert chunk.f3 == "field3_value"
        assert chunk.f4 == "field4_value"
        assert chunk.f5 == "field5_value"

        # Verify combined metadata is preserved
        assert chunk.metadata["_tags"] == "test,chunk,fields"
        assert chunk.metadata["_f1"] == "field1_value"
        assert "_reference_url" in chunk.metadata  # Should be present
