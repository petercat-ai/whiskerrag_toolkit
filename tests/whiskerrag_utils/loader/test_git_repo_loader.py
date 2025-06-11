import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from git import Repo

from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
    Knowledge,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
)
from whiskerrag_types.model.knowledge_source import (
    GithubFileSourceConfig,
    GithubRepoSourceConfig,
)
from whiskerrag_types.model.splitter import GithubRepoParseConfig, TextSplitConfig
from whiskerrag_utils.loader.git_repo_loader import GithubRepoLoader


@pytest.fixture
def sample_repo() -> Any:
    """create a temporary git repo for testing"""
    repo_path = tempfile.mkdtemp()
    repo = Repo.init(repo_path)

    # create test files
    md_content = "# Test Markdown\nThis is a test."
    os.makedirs(os.path.join(repo_path, "docs"))
    with open(os.path.join(repo_path, "docs/test.md"), "w") as f:
        f.write(md_content)

    # add and commit files
    repo.index.add(["docs/test.md"])
    repo.index.commit("Initial commit")

    yield repo_path

    try:
        import shutil

        shutil.rmtree(repo_path)
    except Exception as e:
        print(f"Error cleaning up test repo: {e}")


@pytest.fixture
def mock_knowledge() -> Knowledge:
    """create a test Knowledge instance"""
    return Knowledge(
        knowledge_name="petercat-ai/petercat",
        source_type=KnowledgeSourceEnum.GITHUB_REPO,
        knowledge_type=KnowledgeTypeEnum.GITHUB_REPO,
        source_config=GithubRepoSourceConfig(
            repo_name="petercat-ai/petercat",
            url="https://github.com",
            branch="main",
            auth_info="mock_token",
            commit_id=None,
        ),
        space_id="test_space",
        embedding_model_name=EmbeddingModelEnum.OPENAI,
        split_config={
            "type": "text",
            "chunk_size": 500,
            "chunk_overlap": 100,
            "separators": [],
            "is_separator_regex": True,
        },
        tenant_id="38fbd88b-e869-489c-9142-e4ea2c2261db",
        enabled=True,
    )


class TestGithubRepoLoader:

    @pytest.mark.asyncio
    async def test_initialization(self, mock_knowledge) -> None:
        """test loader initialization"""
        # Mock the lazy import function to return mock git classes
        mock_repo_class = MagicMock()
        mock_clone_from = MagicMock()
        mock_repo_class.clone_from = mock_clone_from

        with patch(
            "whiskerrag_utils.loader.git_repo_loader._lazy_import_git",
            return_value=(mock_repo_class, Exception, Exception),
        ), patch(
            "whiskerrag_utils.loader.git_repo_loader._check_git_installation",
            return_value=True,
        ):
            loader = GithubRepoLoader(mock_knowledge)
            assert loader.repo_name == "petercat-ai/petercat"
            assert loader.branch_name == "main"
            assert loader.token == "mock_token"
            assert mock_clone_from.called

    @pytest.mark.asyncio
    async def test_get_file_list(self, sample_repo, mock_knowledge) -> None:
        """test file list retrieval"""

        def fake_load_repo(self):
            self.repo_path = sample_repo
            self.local_repo = Repo(sample_repo)

        with patch.object(GithubRepoLoader, "_load_repo", fake_load_repo):
            loader = GithubRepoLoader(mock_knowledge)
            file_list = await loader.decompose()

            assert len(file_list) == 1
            assert file_list[0].knowledge_type == KnowledgeTypeEnum.MARKDOWN
            assert "test.md" in file_list[0].knowledge_name

    @pytest.mark.asyncio
    async def test_decompose(self, sample_repo, mock_knowledge) -> None:
        def fake_load_repo(self):
            self.repo_path = sample_repo
            self.local_repo = Repo(sample_repo)

        with patch.object(GithubRepoLoader, "_load_repo", fake_load_repo):
            loader = GithubRepoLoader(mock_knowledge)
            result = await loader.decompose()

            assert len(result) == 1
            assert isinstance(result[0], Knowledge)
            assert result[0].knowledge_type == KnowledgeTypeEnum.MARKDOWN

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_knowledge) -> None:
        # Mock the lazy import function to return mock git classes
        mock_repo_class = MagicMock()
        mock_clone_from = MagicMock()
        mock_repo_class.clone_from = mock_clone_from

        with patch(
            "whiskerrag_utils.loader.git_repo_loader._lazy_import_git",
            return_value=(mock_repo_class, Exception, Exception),
        ), patch(
            "whiskerrag_utils.loader.git_repo_loader._check_git_installation",
            return_value=True,
        ):
            loader = GithubRepoLoader(mock_knowledge)
            repo_path = loader.repo_path

            os.makedirs(repo_path, exist_ok=True)
            assert os.path.exists(repo_path)
            await loader.on_load_finished()
            assert not os.path.exists(repo_path)

    def test_invalid_source_config(self) -> None:
        invalid_knowledge = Knowledge(
            source_type=KnowledgeSourceEnum.GITHUB_FILE,
            knowledge_type=KnowledgeTypeEnum.MARKDOWN,
            source_config=GithubFileSourceConfig(
                repo_name="test/repo",
                url="https://github.com",
                branch="main",
                auth_info="test_token",
                commit_id=None,
                path="docs/test.md",
            ),  # has path but still invalid (e.g. repo does not exist)
            knowledge_name="test",
            space_id="test_space",
            tenant_id="test",
            embedding_model_name=EmbeddingModelEnum.OPENAI,
            split_config=TextSplitConfig(
                type="text",
                chunk_size=500,
                chunk_overlap=100,
                separators=[],
                is_separator_regex=True,
            ),
            enabled=True,
        )

        with pytest.raises(ValueError):
            GithubRepoLoader(invalid_knowledge)

    @pytest.mark.asyncio
    async def test_git_metadata(self, sample_repo, mock_knowledge) -> None:
        def fake_load_repo(self):
            self.repo_path = sample_repo
            self.local_repo = Repo(sample_repo)

        with patch.object(GithubRepoLoader, "_load_repo", fake_load_repo):
            loader = GithubRepoLoader(mock_knowledge)
            file_list = await loader.decompose()

            assert len(file_list) == 1
            metadata = file_list[0].metadata
            assert "_knowledge_type" in metadata

    def test_error_handling(self, mock_knowledge) -> None:
        """test error handling"""
        # Mock the lazy import function and make clone_from raise an exception
        mock_repo_class = MagicMock()
        mock_clone_from = MagicMock(side_effect=Exception("Test error"))
        mock_repo_class.clone_from = mock_clone_from

        with patch(
            "whiskerrag_utils.loader.git_repo_loader._lazy_import_git",
            return_value=(mock_repo_class, Exception, Exception),
        ), patch(
            "whiskerrag_utils.loader.git_repo_loader._check_git_installation",
            return_value=True,
        ):
            with pytest.raises(ValueError) as exc_info:
                GithubRepoLoader(mock_knowledge)
            assert "Failed to load repo" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_pattern_include_and_ignore(
        self, sample_repo, mock_knowledge
    ) -> None:
        """test include_patterns and ignore_patterns priority and effect"""
        # add more files
        with open(os.path.join(sample_repo, "docs/test2.txt"), "w") as f:
            f.write("plain text")
        with open(os.path.join(sample_repo, "docs/ignore.md"), "w") as f:
            f.write("should be ignored")
        with open(os.path.join(sample_repo, "docs/keep.md"), "w") as f:
            f.write("should be kept")

        repo = Repo(sample_repo)
        repo.index.add(["docs/test2.txt", "docs/ignore.md", "docs/keep.md"])
        repo.index.commit("Add more files")

        def fake_load_repo(self):
            self.repo_path = sample_repo
            self.local_repo = Repo(sample_repo)

        # only include .md files, ignore ignore.md
        split_config = GithubRepoParseConfig(
            type="github_repo",
            include_patterns=["*.md"],
            ignore_patterns=["*ignore.md"],
            use_gitignore=True,
            use_default_ignore=True,
        )
        knowledge = mock_knowledge
        knowledge.split_config = split_config

        with patch.object(GithubRepoLoader, "_load_repo", fake_load_repo):
            loader = GithubRepoLoader(knowledge)
            file_list = await loader.decompose()
            names = [f.knowledge_name for f in file_list]
            assert any("test.md" in n for n in names)
            assert any("keep.md" in n for n in names)
            assert not any("ignore.md" in n for n in names)
            assert not any("test2.txt" in n for n in names)

    @pytest.mark.asyncio
    async def test_pattern_no_gitignore_and_default(
        self, sample_repo, mock_knowledge
    ) -> None:
        """test no_gitignore and no_default_ignore_patterns effect"""
        # add .gitignore and default ignore files
        with open(os.path.join(sample_repo, ".gitignore"), "w") as f:
            f.write("*.log\n")
        with open(os.path.join(sample_repo, "docs/should.log"), "w") as f:
            f.write("should be ignored by gitignore")
        with open(os.path.join(sample_repo, "docs/should_keep.log"), "w") as f:
            f.write("should be kept")
        repo = Repo(sample_repo)
        repo.index.add([".gitignore", "docs/should.log", "docs/should_keep.log"])
        repo.index.commit("Add .gitignore and log files")

        def fake_load_repo(self):
            self.repo_path = sample_repo
            self.local_repo = Repo(sample_repo)

        # 不使用 .gitignore 和默认忽略，所有 .log 文件都应被包含
        split_config = GithubRepoParseConfig(
            type="github_repo",
            include_patterns=["*.log"],
            ignore_patterns=[],
            use_gitignore=False,
            use_default_ignore=False,
        )
        knowledge = mock_knowledge
        knowledge.split_config = split_config

        with patch.object(GithubRepoLoader, "_load_repo", fake_load_repo):
            loader = GithubRepoLoader(knowledge)
            file_list = await loader.decompose()
            names = [f.knowledge_name for f in file_list]
            assert any("should.log" in n for n in names)
            assert any("should_keep.log" in n for n in names)

        # use default ignore, all .log files should be excluded
        split_config.use_gitignore = True
        split_config.use_default_ignore = True
        knowledge.split_config = split_config
        with patch.object(GithubRepoLoader, "_load_repo", fake_load_repo):
            loader = GithubRepoLoader(knowledge)
            file_list = await loader.decompose()
            names = [f.knowledge_name for f in file_list]
            assert not any("should.log" in n for n in names)
            assert not any("should_keep.log" in n for n in names)

        # use .gitignore, use_default_ignore
        split_config.use_gitignore = True
        split_config.use_default_ignore = False
        knowledge.split_config = split_config
        with patch.object(GithubRepoLoader, "_load_repo", fake_load_repo):
            loader = GithubRepoLoader(knowledge)
            file_list = await loader.decompose()
            names = [f.knowledge_name for f in file_list]
            assert not any("should.log" in n for n in names)
            assert not any("should_keep.log" in n for n in names)

    @pytest.mark.asyncio
    async def test_real_github_repo_loader(self):
        """real scenario: use real access_token to pull petercat-ai/petercat repo"""
        repo_name = "petercat-ai/petercat"
        knowledge = Knowledge(
            knowledge_name=repo_name,
            source_type=KnowledgeSourceEnum.GITHUB_REPO,
            knowledge_type=KnowledgeTypeEnum.GITHUB_REPO,
            source_config=GithubRepoSourceConfig(
                repo_name=repo_name,
                url="https://github.com/",
                commit_id=None,
            ),
            space_id="test_space",
            embedding_model_name=EmbeddingModelEnum.OPENAI,
            split_config=GithubRepoParseConfig(
                type="github_repo",
                include_patterns=[
                    "*.md",
                    "*.mdx",
                ],
                ignore_patterns=[],
                use_gitignore=True,
                use_default_ignore=True,
            ),
            tenant_id="test_tenant",
            enabled=True,
        )
        loader = GithubRepoLoader(knowledge)
        knowledge_list = await loader.decompose()
        assert len(knowledge_list) > 0
        assert any(
            f.knowledge_type == KnowledgeTypeEnum.MARKDOWN for f in knowledge_list
        )
        await loader.on_load_finished()

    @pytest.mark.asyncio
    async def test_load_returns_tree_and_author(
        self, sample_repo, mock_knowledge
    ) -> None:
        def fake_load_repo(self):
            self.repo_path = sample_repo
            self.local_repo = Repo(sample_repo)

        with patch.object(GithubRepoLoader, "_load_repo", fake_load_repo):
            loader = GithubRepoLoader(mock_knowledge)
            result = await loader.load()
            assert len(result) == 1
            text_obj = result[0]
            assert "docs" in text_obj.content
            assert "test.md" in text_obj.content
            assert "author_name" in text_obj.metadata
            assert "author_email" in text_obj.metadata
