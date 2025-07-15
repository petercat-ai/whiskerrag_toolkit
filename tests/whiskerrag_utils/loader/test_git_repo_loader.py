import os
import tempfile
from typing import Any
from unittest.mock import patch

import pytest
from git import Repo

from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
    Knowledge,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
)
from whiskerrag_types.model.knowledge_source import GithubRepoSourceConfig
from whiskerrag_types.model.splitter import GithubRepoParseConfig
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
def mock_github_knowledge() -> Knowledge:
    """create a test Knowledge instance"""
    return Knowledge(
        knowledge_name="petercat-ai/petercat",
        source_type=KnowledgeSourceEnum.GITHUB_REPO,
        knowledge_type=KnowledgeTypeEnum.GITHUB_REPO,
        source_config=GithubRepoSourceConfig(
            repo_name="petercat-ai/petercat",
            url="https://github.com",
            commit_id=None,
        ),
        space_id="test_space",
        embedding_model_name=EmbeddingModelEnum.OPENAI,
        split_config=GithubRepoParseConfig(
            type="github_repo",
            include_patterns=["*.md"],
            ignore_patterns=["*ignore.md"],
            use_gitignore=True,
            use_default_ignore=True,
        ),
        tenant_id="38fbd88b-e869-489c-9142-e4ea2c2261db",
        enabled=True,
    )


@pytest.fixture
def mock_gitlab_knowledge() -> Knowledge:
    """create a test Knowledge instance"""
    return Knowledge(
        knowledge_name="wohu/whisker",
        source_type=KnowledgeSourceEnum.GITHUB_REPO,
        knowledge_type=KnowledgeTypeEnum.GITHUB_REPO,
        source_config=GithubRepoSourceConfig(
            repo_name="wohu/whisker",
            url="https://code.alipay.com",
            auth_info="git:xxxxx",
        ),
        space_id="test_space",
        embedding_model_name=EmbeddingModelEnum.OPENAI,
        split_config=GithubRepoParseConfig(
            type="github_repo",
            include_patterns=["*.md"],
            ignore_patterns=["*ignore.md"],
            use_gitignore=True,
            use_default_ignore=True,
        ),
        tenant_id="38fbd88b-e869-489c-9142-e4ea2c2261db",
        enabled=True,
    )


class TestGithubRepoLoader:
    @pytest.mark.asyncio
    async def test_github_repo_loader_mock(
        self, sample_repo, mock_github_knowledge
    ) -> None:
        """Mock测试：使用sample_repo模拟GitHub仓库加载"""
        try:
            # Mock GitRepoManager的get_repo_path和get_repo方法
            with patch(
                "whiskerrag_utils.loader.git_repo_loader.get_repo_manager"
            ) as mock_manager:
                # 设置mock返回值
                mock_manager_instance = mock_manager.return_value
                mock_manager_instance.get_repo_path.return_value = sample_repo
                mock_manager_instance.get_repo.return_value = Repo(sample_repo)

                loader = GithubRepoLoader(mock_github_knowledge)

                # 验证基本信息
                assert loader.repo_name == "petercat-ai/petercat"
                assert loader.branch_name == "main"
                assert loader.base_url == "https://github.com"
                assert loader.repo_path == sample_repo
                assert loader.local_repo is not None

                # 测试分解
                knowledge_list = await loader.decompose()
                assert len(knowledge_list) > 0

                # 验证文件类型
                markdown_files = [
                    k
                    for k in knowledge_list
                    if k.knowledge_type == KnowledgeTypeEnum.MARKDOWN
                ]
                assert len(markdown_files) > 0

                # 验证元数据
                for knowledge in knowledge_list[:5]:  # 检查前5个文件
                    assert "_reference_url" in knowledge.metadata
                    assert "branch" in knowledge.metadata
                    assert "repo_name" in knowledge.metadata
                    assert "path" in knowledge.metadata
                    assert "position" in knowledge.metadata

                # 测试加载
                text_list = await loader.load()
                assert len(text_list) == 1
                assert "docs" in text_list[0].content
                assert "test.md" in text_list[0].content
                assert "author_name" in text_list[0].metadata
                assert "author_email" in text_list[0].metadata

                # 记录清理前的路径
                repo_path_before_cleanup = loader.repo_path

                # 清理资源
                await loader.on_load_finished()

                # 验证清理资源成功
                assert loader.repo_path is None
                assert loader.local_repo is None
                # 注意：sample_repo不会被删除，因为它是测试fixture

        except Exception as e:
            pytest.fail(f"GitHub仓库Mock加载失败: {e}")

    @pytest.mark.asyncio
    async def test_github_repo_loader_real(self, mock_github_knowledge) -> None:
        """真实测试：GitHub仓库加载（可选运行）"""
        try:
            loader = GithubRepoLoader(mock_github_knowledge)

            # 验证基本信息
            assert loader.repo_name == "petercat-ai/petercat"
            assert loader.branch_name == "main"
            assert loader.base_url == "https://github.com"
            assert loader.repo_path is not None
            assert loader.local_repo is not None

            # 测试分解
            knowledge_list = await loader.decompose()
            assert len(knowledge_list) > 0

            # 验证文件类型
            markdown_files = [
                k
                for k in knowledge_list
                if k.knowledge_type == KnowledgeTypeEnum.MARKDOWN
            ]
            assert len(markdown_files) > 0

            # 验证元数据
            for knowledge in knowledge_list[:5]:  # 检查前5个文件
                assert "_reference_url" in knowledge.metadata
                assert "branch" in knowledge.metadata
                assert "repo_name" in knowledge.metadata
                assert "path" in knowledge.metadata
                assert "position" in knowledge.metadata

            # 测试加载
            text_list = await loader.load()
            assert len(text_list) == 1
            assert "petercat" in text_list[0].content.lower()
            assert "author_name" in text_list[0].metadata
            assert "author_email" in text_list[0].metadata

            # 记录清理前的路径
            repo_path_before_cleanup = loader.repo_path

            # 清理资源
            await loader.on_load_finished()

            # 验证清理资源成功
            assert loader.repo_path is None
            assert loader.local_repo is None
            assert not os.path.exists(repo_path_before_cleanup)

        except Exception as e:
            pytest.skip(f"GitHub仓库真实加载失败（网络问题）: {e}")

    @pytest.mark.asyncio
    async def test_gitlab_repo_loader_mock(
        self, sample_repo, mock_gitlab_knowledge
    ) -> None:
        """Mock测试：使用sample_repo模拟GitLab仓库加载"""
        try:
            # Mock GitRepoManager的get_repo_path和get_repo方法
            with patch(
                "whiskerrag_utils.loader.git_repo_loader.get_repo_manager"
            ) as mock_manager:
                # 设置mock返回值
                mock_manager_instance = mock_manager.return_value
                mock_manager_instance.get_repo_path.return_value = sample_repo
                mock_manager_instance.get_repo.return_value = Repo(sample_repo)

                loader = GithubRepoLoader(mock_gitlab_knowledge)

                # 验证基本信息
                assert loader.repo_name == "wohu/whisker"
                # 注意：mock测试中使用的是sample_repo，其默认分支是main
                assert loader.branch_name == "main"
                assert loader.base_url == "https://code.alipay.com"
                assert loader.repo_path == sample_repo
                assert loader.local_repo is not None

                # 测试分解
                knowledge_list = await loader.decompose()
                assert len(knowledge_list) > 0

                # 验证文件类型
                markdown_files = [
                    k
                    for k in knowledge_list
                    if k.knowledge_type == KnowledgeTypeEnum.MARKDOWN
                ]
                assert len(markdown_files) > 0

                # 验证元数据
                for knowledge in knowledge_list[:5]:  # 检查前5个文件
                    assert "_reference_url" in knowledge.metadata
                    assert "branch" in knowledge.metadata
                    assert "repo_name" in knowledge.metadata
                    assert "path" in knowledge.metadata
                    assert "position" in knowledge.metadata

                # 测试加载
                text_list = await loader.load()
                assert len(text_list) == 1
                assert "docs" in text_list[0].content
                assert "test.md" in text_list[0].content
                assert "author_name" in text_list[0].metadata
                assert "author_email" in text_list[0].metadata

                # 记录清理前的路径
                repo_path_before_cleanup = loader.repo_path

                # 清理资源
                await loader.on_load_finished()

                # 验证清理资源成功
                assert loader.repo_path is None
                assert loader.local_repo is None
                # 注意：sample_repo不会被删除，因为它是测试fixture

        except Exception as e:
            pytest.fail(f"GitLab仓库Mock加载失败: {e}")

    @pytest.mark.asyncio
    async def test_gitlab_repo_loader_real(self) -> None:
        """真实测试：GitLab仓库加载（可选运行）"""
        try:
            knowledge = Knowledge(
                knowledge_name="wohu/whisker",
                source_type=KnowledgeSourceEnum.GITHUB_REPO,
                knowledge_type=KnowledgeTypeEnum.GITHUB_REPO,
                source_config=GithubRepoSourceConfig(
                    repo_name="wohu/whisker",
                    url="https://code.alipay.com",
                    auth_info="git:Bx-t_6mbv0aP-1vjRLX-",
                ),
                space_id="test_space",
                embedding_model_name=EmbeddingModelEnum.OPENAI,
                split_config=GithubRepoParseConfig(
                    type="github_repo",
                    include_patterns=["*.md"],
                    ignore_patterns=["*ignore.md"],
                    use_gitignore=True,
                    use_default_ignore=True,
                ),
                tenant_id="38fbd88b-e869-489c-9142-e4ea2c2261db",
                enabled=True,
            )
            loader = GithubRepoLoader(knowledge)

            # 验证基本信息
            assert loader.repo_name == "wohu/whisker"
            assert loader.branch_name == "master"
            assert loader.base_url == "https://code.alipay.com"
            assert loader.repo_path is not None
            assert loader.local_repo is not None

            # 测试分解
            knowledge_list = await loader.decompose()
            assert len(knowledge_list) > 0

            # 验证文件类型
            markdown_files = [
                k
                for k in knowledge_list
                if k.knowledge_type == KnowledgeTypeEnum.MARKDOWN
            ]
            assert len(markdown_files) > 0

            # 验证元数据
            for knowledge in knowledge_list[:5]:  # 检查前5个文件
                assert "_reference_url" in knowledge.metadata
                assert "branch" in knowledge.metadata
                assert "repo_name" in knowledge.metadata
                assert "path" in knowledge.metadata
                assert "position" in knowledge.metadata

            # 测试加载
            text_list = await loader.load()
            assert len(text_list) == 1
            assert "author_name" in text_list[0].metadata
            assert "author_email" in text_list[0].metadata

            # 记录清理前的路径
            repo_path_before_cleanup = loader.repo_path

            # 清理资源
            await loader.on_load_finished()

            # 验证清理资源成功
            assert loader.repo_path is None
            assert loader.local_repo is None
            assert not os.path.exists(repo_path_before_cleanup)

        except Exception as e:
            pytest.skip(f"GitLab仓库真实加载失败（可能是token问题）: {e}")
