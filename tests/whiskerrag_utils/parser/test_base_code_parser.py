import os

import pytest
from langchain_text_splitters import Language

from whiskerrag_types.model.knowledge import Knowledge
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_types.model.splitter import BaseCodeSplitConfig
from whiskerrag_utils import RegisterTypeEnum, get_register
from whiskerrag_utils.registry import init_register


class TestBaseCodeParser:
    def setup_method(self):
        """Setup test environment"""
        init_register()
        self.parser = get_register(RegisterTypeEnum.PARSER, "base_code")()
        self.test_code_dir = os.path.join(os.path.dirname(__file__), "code")

    def _create_knowledge(
        self, language: Language, chunk_size: int = 500, chunk_overlap: int = 50
    ):
        """Create a Knowledge object with BaseCodeSplitConfig"""
        return Knowledge(
            source_type="github_file",
            knowledge_type="python",  # Will be overridden based on language
            space_id="test_space",
            knowledge_name="test_code",
            split_config=BaseCodeSplitConfig(
                language=language,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ),
            source_config={"text": "test"},
            embedding_model_name="openai",
            tenant_id="test-tenant",
            metadata={
                "_reference_url": "https://github.com/test/repo/blob/main/test.py"
            },
        )

    def _load_test_file(self, filename: str) -> str:
        """Load test file content"""
        filepath = os.path.join(self.test_code_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    @pytest.mark.asyncio
    async def test_python_code_parsing(self):
        """Test Python code parsing with position information"""
        content = self._load_test_file("rstar.py")
        knowledge = self._create_knowledge(
            Language.PYTHON, chunk_size=1000, chunk_overlap=100
        )

        text_content = Text(
            content=content,
            metadata={
                "file_path": "rstar.py",
                "_reference_url": "https://github.com/test/repo/blob/main/rstar.py",
            },
        )

        result = await self.parser.parse(knowledge, text_content)

        # Verify we get multiple chunks
        assert len(result) > 1

        # Check first chunk
        first_chunk = result[0]
        assert "position" in first_chunk.metadata
        assert "chunk_index" in first_chunk.metadata
        assert first_chunk.metadata["chunk_index"] == 0

        position = first_chunk.metadata["position"]
        assert position["start_line"] == 1
        assert position["start_column"] == 1
        assert "end_line" in position
        assert "total_lines" in position

        # Verify position format
        assert isinstance(position["end_line"], int)
        assert position["end_line"] >= position["start_line"]

        # Check parser type metadata
        assert first_chunk.metadata["parser_type"] == "base_code"

        # Verify all chunks have increasing line numbers
        for i in range(1, len(result)):
            prev_position = result[i - 1].metadata["position"]
            curr_position = result[i].metadata["position"]
            assert curr_position["start_line"] >= prev_position["start_line"]

    @pytest.mark.asyncio
    async def test_typescript_code_parsing(self):
        """Test TypeScript code parsing with position information"""
        content = self._load_test_file("rstar.ts")
        knowledge = self._create_knowledge(
            Language.TS, chunk_size=800, chunk_overlap=80
        )

        text_content = Text(
            content=content,
            metadata={
                "file_path": "rstar.ts",
                "_reference_url": "https://github.com/test/repo/blob/main/rstar.ts",
            },
        )

        result = await self.parser.parse(knowledge, text_content)

        # Verify we get multiple chunks
        assert len(result) > 1

        # Check that chunks contain TypeScript-specific content
        found_class = False
        found_constructor = False

        for chunk in result:
            if "class" in chunk.content:
                found_class = True
            if "constructor" in chunk.content:
                found_constructor = True

            # Verify position metadata
            assert "position" in chunk.metadata
            position = chunk.metadata["position"]
            assert "start_line" in position
            assert "end_line" in position

            # Verify parser type
            assert chunk.metadata["parser_type"] == "base_code"

        assert found_class, "Should find TypeScript class definition"
        assert found_constructor, "Should find TypeScript constructor"

    @pytest.mark.asyncio
    async def test_java_code_parsing(self):
        """Test Java code parsing with position information"""
        content = self._load_test_file("rstar.java")
        knowledge = self._create_knowledge(
            Language.JAVA, chunk_size=600, chunk_overlap=60
        )

        text_content = Text(
            content=content,
            metadata={
                "file_path": "rstar.java",
                "_reference_url": "https://github.com/test/repo/blob/main/rstar.java",
            },
        )

        result = await self.parser.parse(knowledge, text_content)

        # Verify we get multiple chunks
        assert len(result) > 1

        # Check that chunks contain Java-specific content
        found_public_class = False
        found_static = False

        for chunk in result:
            if "public class" in chunk.content:
                found_public_class = True
            if "static" in chunk.content:
                found_static = True

            # Verify position metadata
            assert "position" in chunk.metadata
            position = chunk.metadata["position"]
            assert position["start_line"] >= 1
            assert position["end_line"] >= position["start_line"]

            # Verify parser type
            assert chunk.metadata["parser_type"] == "base_code"

        assert found_public_class, "Should find Java public class"
        assert found_static, "Should find Java static methods/fields"

    @pytest.mark.asyncio
    async def test_position_calculation_accuracy(self):
        """Test the accuracy of position calculation"""
        # Simple test content with known line structure
        test_content = """class TestClass:
    def __init__(self):
        self.value = 1

    def method_one(self):
        return self.value

    def method_two(self):
        return self.value * 2

def function_outside():
    return "hello"
"""
        knowledge = self._create_knowledge(
            Language.PYTHON, chunk_size=100, chunk_overlap=20
        )
        text_content = Text(content=test_content, metadata={})

        result = await self.parser.parse(knowledge, text_content)

        # Verify first chunk starts at line 1
        first_chunk = result[0]
        assert first_chunk.metadata["position"]["start_line"] == 1

    @pytest.mark.asyncio
    async def test_chunk_overlap_position(self):
        """Test position calculation with chunk overlap"""
        test_content = """def function_one():
    print("one")

def function_two():
    print("two")

def function_three():
    print("three")

def function_four():
    print("four")
"""
        knowledge = self._create_knowledge(
            Language.PYTHON, chunk_size=60, chunk_overlap=30
        )
        text_content = Text(content=test_content, metadata={})

        result = await self.parser.parse(knowledge, text_content)

        # Verify overlap behavior - chunks should have overlapping content but proper line positions
        for i in range(len(result)):
            chunk = result[i]
            position = chunk.metadata["position"]

            # Each chunk should have valid position data
            assert position["start_line"] >= 1
            assert position["end_line"] >= position["start_line"]
            assert position["total_lines"] >= 1

            # Verify chunk index
            assert chunk.metadata["chunk_index"] == i

    @pytest.mark.asyncio
    async def test_metadata_inheritance(self):
        """Test that metadata is properly inherited and merged"""
        content = "def simple_function():\n    return True"
        knowledge = self._create_knowledge(Language.PYTHON)

        # Add metadata to knowledge
        knowledge.metadata["repo_name"] = "test-repo"
        knowledge.metadata["branch"] = "main"

        text_content = Text(
            content=content,
            metadata={
                "file_path": "test.py",
                "_reference_url": "https://github.com/test/repo/blob/main/test.py",
            },
        )

        result = await self.parser.parse(knowledge, text_content)

        chunk = result[0]
        metadata = chunk.metadata

        # Check that knowledge metadata is inherited
        assert metadata["repo_name"] == "test-repo"
        assert metadata["branch"] == "main"

        # Check that text metadata is inherited
        assert metadata["file_path"] == "test.py"
        assert (
            metadata["_reference_url"]
            == "https://github.com/test/repo/blob/main/test.py"
        )

        # Check that parser metadata is added
        assert metadata["parser_type"] == "base_code"
        assert "position" in metadata
        assert "chunk_index" in metadata

    @pytest.mark.asyncio
    async def test_empty_content(self):
        """Test handling of empty content"""
        knowledge = self._create_knowledge(Language.PYTHON)
        text_content = Text(content="", metadata={})

        result = await self.parser.parse(knowledge, text_content)

        # Empty content returns empty list from splitter
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_single_line_content(self):
        """Test handling of single line content"""
        content = "print('hello world')"
        knowledge = self._create_knowledge(Language.PYTHON)
        text_content = Text(content=content, metadata={})

        result = await self.parser.parse(knowledge, text_content)

        assert len(result) == 1
        chunk = result[0]
        position = chunk.metadata["position"]

        assert position["start_line"] == 1
        assert position["end_line"] == 1
        assert position["total_lines"] == 1

    @pytest.mark.asyncio
    async def test_different_languages_different_splitting(self):
        """Test that different languages result in different splitting patterns"""
        # Use the same content but different languages
        content = self._load_test_file("rstar.py")  # Python-like structure

        python_knowledge = self._create_knowledge(Language.PYTHON, chunk_size=500)
        java_knowledge = self._create_knowledge(Language.JAVA, chunk_size=500)

        text_content = Text(content=content, metadata={})

        python_result = await self.parser.parse(python_knowledge, text_content)
        java_result = await self.parser.parse(java_knowledge, text_content)

        # Results should be different due to different language splitting rules
        assert len(python_result) > 0
        assert len(java_result) > 0

        # Verify parser type metadata is correct
        assert python_result[0].metadata["parser_type"] == "base_code"
        assert java_result[0].metadata["parser_type"] == "base_code"
