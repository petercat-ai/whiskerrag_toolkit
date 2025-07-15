import os
from unittest.mock import patch

import pytest

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
from whiskerrag_types.model.splitter import GithubRepoParseConfig
from whiskerrag_utils.loader.git_file_loader import GithubFileLoader


@pytest.fixture
def mock_github_file_knowledge() -> Knowledge:
    """create a test Knowledge instance for GitHub file loading"""
    return Knowledge(
        knowledge_name="test-file",
        source_type=KnowledgeSourceEnum.GITHUB_FILE,
        knowledge_type=KnowledgeTypeEnum.MARKDOWN,
        source_config=GithubFileSourceConfig(
            repo_name="petercat-ai/petercat",
            url="https://github.com",
            branch="main",
            auth_info=None,
            commit_id=None,
            path="README.md",
        ),
        space_id="test_space",
        embedding_model_name=EmbeddingModelEnum.OPENAI,
        split_config={
            "type": "markdown",
            "chunk_size": 500,
            "chunk_overlap": 100,
            "separators": [],
            "is_separator_regex": True,
        },
        tenant_id="38fbd88b-e869-489c-9142-e4ea2c2261db",
        enabled=True,
    )


@pytest.fixture
def mock_gitlab_file_knowledge() -> Knowledge:
    """create a test Knowledge instance for GitLab file loading"""
    return Knowledge(
        knowledge_name="test-gitlab-file",
        source_type=KnowledgeSourceEnum.GITHUB_FILE,
        knowledge_type=KnowledgeTypeEnum.MARKDOWN,
        source_config=GithubFileSourceConfig(
            repo_name="wohu/whisker",
            url="https://code.alipay.com",
            branch="master",
            auth_info="git:xxxxx",
            commit_id=None,
            path="README.md",
        ),
        space_id="test_space",
        embedding_model_name=EmbeddingModelEnum.OPENAI,
        split_config={
            "type": "markdown",
            "chunk_size": 500,
            "chunk_overlap": 100,
            "separators": [],
            "is_separator_regex": True,
        },
        tenant_id="38fbd88b-e869-489c-9142-e4ea2c2261db",
        enabled=True,
    )


@pytest.fixture
def sample_repo():
    import shutil
    import tempfile

    from git import Repo

    repo_path = tempfile.mkdtemp()
    repo = Repo.init(repo_path)
    md_content = "# Test Markdown\nThis is a test file for GitHub file loader."
    with open(f"{repo_path}/README.md", "w") as f:
        f.write(md_content)
    repo.index.add(["README.md"])
    repo.index.commit("init")
    yield repo_path
    shutil.rmtree(repo_path)


@pytest.mark.asyncio
async def test_github_file_loader_mock(sample_repo, mock_github_file_knowledge):
    with patch(
        "whiskerrag_utils.loader.git_file_loader.get_repo_manager"
    ) as mock_manager:
        mock_manager_instance = mock_manager.return_value
        mock_manager_instance.get_repo_path.return_value = sample_repo
        loader = GithubFileLoader(mock_github_file_knowledge)
        result = await loader.load()
        assert len(result) == 1
        assert "Test Markdown" in result[0].content


@pytest.mark.asyncio
async def test_gitlab_file_loader_mock(sample_repo, mock_gitlab_file_knowledge):
    with patch(
        "whiskerrag_utils.loader.git_file_loader.get_repo_manager"
    ) as mock_manager:
        mock_manager_instance = mock_manager.return_value
        mock_manager_instance.get_repo_path.return_value = sample_repo
        loader = GithubFileLoader(mock_gitlab_file_knowledge)
        result = await loader.load()
        assert len(result) == 1
        assert "Test Markdown" in result[0].content
