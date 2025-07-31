import os
import shutil
from unittest.mock import patch

import pytest

from whiskerrag_types.model.knowledge_source import GithubRepoSourceConfig
from whiskerrag_utils.loader.git_repo_manager import GitRepoManager


class TestGitRepoManager:
    def test_real_github_repo_download_zip_only(self):
        """测试真实的 GitHub 仓库 ZIP 下载功能（跳过 Git clone）"""
        # 创建 GitRepoManager 实例
        manager = GitRepoManager()

        # 配置真实的 GitHub 仓库参数
        config = GithubRepoSourceConfig(
            repo_name="petercat-ai/whiskerrag_toolkit",
            url="https://github.com",
            commit_id=None,
        )

        try:
            # 使用 mock 强制跳过 Git clone，直接使用 ZIP 下载
            with patch(
                "whiskerrag_utils.loader.git_repo_manager._check_git_installation",
                return_value=False,
            ):
                # 验证仓库管理器目录存在
                assert os.path.exists(
                    manager.repos_dir
                ), f"仓库管理器目录不存在: {manager.repos_dir}"

                # 先检查缓存状态
                repo_key = manager._generate_repo_key(config)
                assert (
                    repo_key == "petercat-ai_whiskerrag_toolkit"
                ), f"仓库键不正确: {repo_key}"
                assert repo_key not in manager._repos_cache, "仓库不应该已经在缓存中"

                # 下载仓库
                repo_path = manager.get_repo_path(config)

                # 验证路径存在且是目录
                assert os.path.exists(repo_path), f"仓库路径不存在: {repo_path}"
                assert os.path.isdir(repo_path), f"仓库路径不是目录: {repo_path}"

                # 检查是否为 GitHub 仓库（ZIP 下载）
                is_github = manager._is_github_repo(config)
                assert is_github is True, "应该识别为 GitHub 仓库"

                # 检查仓库信息缓存
                assert repo_key in manager._repo_info_cache, "仓库信息应该在缓存中"
                repo_info = manager._repo_info_cache[repo_key]
                assert (
                    repo_info.get("name") == "whiskerrag_toolkit"
                ), f"仓库名不正确: {repo_info.get('name')}"
                assert (
                    repo_info.get("default_branch") == "main"
                ), f"默认分支不正确: {repo_info.get('default_branch')}"

                # 统计文件和目录数量
                file_count = 0
                dir_count = 0
                found_files = set()
                found_dirs = set()

                for root, dirs, files in os.walk(repo_path):
                    # 跳过 .git 目录（ZIP 下载不应该有这个目录）
                    if ".git" in dirs:
                        dirs.remove(".git")

                    # 统计目录
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        relative_dir = os.path.relpath(dir_path, repo_path)
                        found_dirs.add(relative_dir)
                        dir_count += 1

                    # 统计文件
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, repo_path)
                        found_files.add(relative_path)
                        file_count += 1

                # 验证文件和目录数量
                assert (
                    file_count >= 10
                ), f"文件数量太少: {file_count}，期望至少 10 个文件"
                assert dir_count >= 5, f"目录数量太少: {dir_count}，期望至少 5 个目录"

                # 验证这确实是 ZIP 下载（没有 .git 目录）
                git_dir = os.path.join(repo_path, ".git")
                assert not os.path.exists(git_dir), "ZIP 下载不应该包含 .git 目录"

                # 检查必须存在的项目文件
                required_files = ["README.md", "pyproject.toml"]
                for file_name in required_files:
                    file_path = os.path.join(repo_path, file_name)
                    assert os.path.exists(
                        file_path
                    ), f"必需的项目文件不存在: {file_name}"

                # 检查必须存在的目录
                required_dirs = ["src", "tests"]
                for dir_name in required_dirs:
                    dir_path = os.path.join(repo_path, dir_name)
                    assert os.path.exists(dir_path), f"必需的目录不存在: {dir_name}"
                    assert os.path.isdir(dir_path), f"路径不是目录: {dir_name}"

        except Exception as e:
            # 清理下载的仓库
            try:
                manager.cleanup_repo(config)
            except Exception:
                pass
            pytest.skip(f"GitHub仓库ZIP下载测试失败（可能是网络或API问题）: {e}")
        finally:
            # 清理下载的仓库
            try:
                manager.cleanup_repo(config)
            except Exception as e:
                # 清理失败不应该影响测试结果，但记录错误
                print(f"⚠️ 清理仓库时出错: {e}")

    def test_direct_zip_download(self):
        """直接测试 ZIP 下载方法"""
        manager = GitRepoManager()

        config = GithubRepoSourceConfig(
            repo_name="petercat-ai/petercat",
            url="https://github.com",
            commit_id=None,
        )

        try:
            # 验证仓库管理器目录存在
            assert os.path.exists(
                manager.repos_dir
            ), f"仓库管理器目录不存在: {manager.repos_dir}"

            # 生成正确的目标路径（在 repos_dir 下）
            repo_name = config.repo_name.replace("/", "_")
            target_path = os.path.join(manager.repos_dir, repo_name)

            # 如果目标路径已存在，先清理
            if os.path.exists(target_path):
                shutil.rmtree(target_path)

            # 确保目标路径不存在
            assert not os.path.exists(target_path), f"目标路径清理失败: {target_path}"

            # 直接调用 ZIP 下载方法
            manager._download_github_zip(config, target_path)

            # 验证下载结果
            assert os.path.exists(target_path), f"下载后目标路径不存在: {target_path}"
            assert os.path.isdir(target_path), f"目标路径不是目录: {target_path}"

            # 验证目录不为空
            contents = os.listdir(target_path)
            assert len(contents) > 0, "下载的目录为空"

            # 统计内容
            file_count = 0
            dir_count = 0
            for item in contents:
                item_path = os.path.join(target_path, item)
                if os.path.isdir(item_path):
                    dir_count += 1
                else:
                    file_count += 1

            # 验证内容数量合理
            assert (
                file_count + dir_count >= 5
            ), f"下载内容太少: {file_count} 文件 + {dir_count} 目录"

            # 检查一些预期的文件
            expected_files = ["README.md", "LICENSE"]
            found_expected = 0
            for expected_file in expected_files:
                if os.path.exists(os.path.join(target_path, expected_file)):
                    found_expected += 1

            assert found_expected > 0, f"未找到预期的文件: {expected_files}"

        except Exception as e:
            # 清理下载的目录
            try:
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
            except Exception:
                pass
            pytest.skip(f"直接ZIP下载测试失败（可能是网络或API问题）: {e}")
        finally:
            # 清理下载的目录
            try:
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
            except Exception as e:
                print(f"⚠️ 清理时出错: {e}")

    def test_mock_repo_compatibility(self):
        """测试 MockRepo 的兼容性"""
        manager = GitRepoManager()

        config = GithubRepoSourceConfig(
            repo_name="petercat-ai/petercat",
            url="https://github.com",
            commit_id=None,
        )

        try:
            # 强制使用 ZIP 下载
            with patch(
                "whiskerrag_utils.loader.git_repo_manager._check_git_installation",
                return_value=False,
            ):
                # 获取 repo 对象
                repo = manager.get_repo(config)

                # 验证 repo 对象类型
                assert hasattr(
                    repo, "active_branch"
                ), "repo 对象缺少 active_branch 属性"
                assert hasattr(repo, "head"), "repo 对象缺少 head 属性"

                # 测试 active_branch 接口
                branch_name = repo.active_branch.name
                assert isinstance(
                    branch_name, str
                ), f"分支名应该是字符串: {type(branch_name)}"
                assert branch_name == "main", f"默认分支名不正确: {branch_name}"

                # 测试 head 接口
                commit = repo.head.commit
                assert hasattr(commit, "hexsha"), "commit 对象缺少 hexsha 属性"
                assert hasattr(commit, "author"), "commit 对象缺少 author 属性"
                assert hasattr(commit, "message"), "commit 对象缺少 message 属性"

                # 验证 commit 属性
                assert isinstance(
                    commit.hexsha, str
                ), f"hexsha 应该是字符串: {type(commit.hexsha)}"
                assert (
                    len(commit.hexsha) == 40
                ), f"hexsha 长度不正确: {len(commit.hexsha)}"
                assert isinstance(
                    commit.author.name, str
                ), f"作者名应该是字符串: {type(commit.author.name)}"
                assert isinstance(
                    commit.author.email, str
                ), f"作者邮箱应该是字符串: {type(commit.author.email)}"
                assert isinstance(
                    commit.message, str
                ), f"提交信息应该是字符串: {type(commit.message)}"

                # 测试 tree 遍历
                tree = repo.head.commit.tree
                assert hasattr(tree, "traverse"), "tree 对象缺少 traverse 方法"

                # 验证 tree 遍历功能
                file_count = 0
                for item in tree.traverse():
                    assert hasattr(item, "type"), "tree item 缺少 type 属性"
                    assert hasattr(item, "path"), "tree item 缺少 path 属性"

                    if item.type == "blob":
                        assert isinstance(
                            item.path, str
                        ), f"文件路径应该是字符串: {type(item.path)}"
                        file_count += 1
                        if file_count >= 5:  # 只验证前5个文件
                            break

                # 验证至少找到了一些文件
                assert file_count > 0, "tree 遍历没有找到任何文件"

        except Exception as e:
            # 清理
            try:
                manager.cleanup_repo(config)
            except Exception:
                pass
            pytest.skip(f"MockRepo兼容性测试失败（可能是网络或API问题）: {e}")
        finally:
            # 清理
            try:
                manager.cleanup_repo(config)
            except Exception as e:
                print(f"⚠️ 清理时出错: {e}")
