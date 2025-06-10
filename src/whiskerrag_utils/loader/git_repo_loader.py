import logging
import os
import shutil
import subprocess
from typing import Dict, List, Optional
from urllib.parse import urlparse

from git import Repo
from git.exc import GitCommandNotFound, InvalidGitRepositoryError
from openai import BaseModel

from whiskerrag_types.interface.loader_interface import BaseLoader
from whiskerrag_types.model.knowledge import (
    Knowledge,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
)
from whiskerrag_types.model.knowledge_source import GithubRepoSourceConfig
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils.loader.file_pattern_manager import FilePatternManager
from whiskerrag_utils.registry import RegisterTypeEnum, register

logger = logging.getLogger(__name__)


class GitFileElementType(BaseModel):
    content: str
    path: str
    mode: str
    url: str
    branch: str
    repo_name: str
    size: int
    sha: str


def _check_git_installation() -> bool:
    """检查 git 是否安装"""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True, text=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _get_temp_git_env() -> Dict[str, str]:
    """
    获取临时的 git 环境配置

    Returns:
        Dict[str, str]: git config
    """
    return {
        "GIT_AUTHOR_NAME": "temp",
        "GIT_AUTHOR_EMAIL": "temp@example.com",
        "GIT_COMMITTER_NAME": "temp",
        "GIT_COMMITTER_EMAIL": "temp@example.com",
        "GIT_TERMINAL_PROMPT": "0",  # disable interactive prompt
    }


@register(RegisterTypeEnum.KNOWLEDGE_LOADER, KnowledgeSourceEnum.GITHUB_REPO)
class GithubRepoLoader(BaseLoader):
    repo_name: str
    branch_name: Optional[str] = None
    token: Optional[str] = None
    local_repo: Optional[Repo] = None
    knowledge: Knowledge

    @property
    def repos_dir(self) -> str:
        """get runtime root folder path"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repos_dir = os.path.join(current_dir, "repo_download")
        return repos_dir

    def __init__(
        self,
        knowledge: Knowledge,
    ):
        """
        init GithubRepoLoader

        Args:
            knowledge: Knowledge instance,which include repo source config

        Raises:
            ValueError: invalid repo config
        """
        if not isinstance(knowledge.source_config, GithubRepoSourceConfig):
            raise ValueError("source_config should be GithubRepoSourceConfig")

        self.knowledge = knowledge
        self.repo_name = knowledge.source_config.repo_name
        self.branch_name = knowledge.source_config.branch
        self.token = knowledge.source_config.auth_info
        self.base_url = knowledge.source_config.url.rstrip("/")

        os.makedirs(self.repos_dir, exist_ok=True)

        self.repo_path: Optional[str] = os.path.join(
            self.repos_dir, self.repo_name.replace("/", "_")
        )

        try:
            self._load_repo()
        except Exception as e:
            logger.error(f"Failed to load repo: {e}")
            raise ValueError(
                f"Failed to load repo {self.repo_name} with branch {self.branch_name}. Error: {str(e)}"
            )

    def _load_repo(self) -> None:
        if not _check_git_installation():
            raise ValueError("Git is not installed in the system")

        if not self.repo_path:
            raise ValueError("Repository path not initialized")

        try:
            parsed_url = urlparse(self.base_url)
            if not parsed_url.scheme or parsed_url.scheme != "https":
                raise ValueError(
                    f"Invalid URL scheme: {self.base_url}. Only HTTPS URLs are supported"
                )

            clone_url = f"{self.base_url}/{self.repo_name}.git"

            git_env = _get_temp_git_env()

            if self.token:
                git_env.update(
                    {
                        "GIT_ASKPASS": "echo",
                        "GIT_USERNAME": "git",
                        "GIT_PASSWORD": self.token,
                    }
                )

            if os.path.exists(self.repo_path):
                try:
                    self.local_repo = Repo(self.repo_path)
                    self._update_repo()
                except Exception as e:
                    logger.warning(f"Failed to update existing repo: {e}")
                    shutil.rmtree(self.repo_path)
                    self._clone_repo(clone_url, git_env)
            else:
                self._clone_repo(clone_url, git_env)

            if not self.branch_name and self.local_repo:
                try:
                    self.branch_name = self.local_repo.active_branch.name
                except Exception as e:
                    logger.warning(f"Failed to get default branch name: {str(e)}")
                    self.branch_name = "main"

            logger.info(f"Successfully loaded repository at {self.repo_path}")

        except Exception as e:
            raise ValueError(
                f"Failed to load repo {self.repo_name} with branch {self.branch_name}. "
                f"Error: {str(e)}"
            )

    def _clone_repo(self, clone_url: str, git_env: Dict[str, str]) -> None:
        if not self.repo_path:
            raise ValueError("Repository path not initialized")

        try:
            if self.branch_name:
                self.local_repo = Repo.clone_from(
                    url=clone_url,
                    to_path=self.repo_path,
                    depth=1,
                    single_branch=True,
                    env=git_env,
                    branch=self.branch_name,
                )
            else:
                self.local_repo = Repo.clone_from(
                    url=clone_url,
                    to_path=self.repo_path,
                    depth=1,
                    single_branch=True,
                    env=git_env,
                )
        except GitCommandNotFound:
            raise ValueError("Git command not found. Please ensure git is installed.")
        except InvalidGitRepositoryError as e:
            raise ValueError(f"Invalid git repository: {str(e)}")
        except Exception as e:
            if "Authentication failed" in str(e):
                raise ValueError("Authentication failed. Please check your token.")
            raise

    def _update_repo(self) -> None:
        if not self.local_repo:
            raise ValueError("Repository not initialized")

        if self.branch_name:
            self.local_repo.git.checkout(self.branch_name)
        self.local_repo.remotes.origin.pull()

    @staticmethod
    def get_knowledge_type_by_ext(ext: str) -> KnowledgeTypeEnum:
        ext = ext.lower()
        ext_to_type = {
            ".md": KnowledgeTypeEnum.MARKDOWN,
            ".mdx": KnowledgeTypeEnum.MARKDOWN,
            ".txt": KnowledgeTypeEnum.TEXT,
            ".json": KnowledgeTypeEnum.JSON,
            ".pdf": KnowledgeTypeEnum.PDF,
            ".docx": KnowledgeTypeEnum.DOCX,
            ".rst": KnowledgeTypeEnum.RST,
            ".py": KnowledgeTypeEnum.PYTHON,
            ".js": KnowledgeTypeEnum.JS,
            ".ts": KnowledgeTypeEnum.TS,
            ".go": KnowledgeTypeEnum.GO,
            ".java": KnowledgeTypeEnum.JAVA,
            ".cpp": KnowledgeTypeEnum.CPP,
            ".c": KnowledgeTypeEnum.C,
            ".h": KnowledgeTypeEnum.C,
            ".hpp": KnowledgeTypeEnum.CPP,
            ".cs": KnowledgeTypeEnum.CSHARP,
            ".kt": KnowledgeTypeEnum.KOTLIN,
            ".swift": KnowledgeTypeEnum.SWIFT,
            ".php": KnowledgeTypeEnum.PHP,
            ".rb": KnowledgeTypeEnum.RUBY,
            ".rs": KnowledgeTypeEnum.RUST,
            ".scala": KnowledgeTypeEnum.SCALA,
            ".sol": KnowledgeTypeEnum.SOL,
            ".html": KnowledgeTypeEnum.HTML,
            ".css": KnowledgeTypeEnum.TEXT,
            ".lua": KnowledgeTypeEnum.LUA,
            ".m": KnowledgeTypeEnum.TEXT,  # Objective-C/MATLAB等
            ".sh": KnowledgeTypeEnum.TEXT,
            ".yml": KnowledgeTypeEnum.TEXT,
            ".yaml": KnowledgeTypeEnum.TEXT,
            ".tex": KnowledgeTypeEnum.LATEX,
            ".jpg": KnowledgeTypeEnum.IMAGE,
            ".jpeg": KnowledgeTypeEnum.IMAGE,
            ".png": KnowledgeTypeEnum.IMAGE,
            ".gif": KnowledgeTypeEnum.IMAGE,
            ".bmp": KnowledgeTypeEnum.IMAGE,
            ".svg": KnowledgeTypeEnum.IMAGE,
        }
        return ext_to_type.get(ext, KnowledgeTypeEnum.TEXT)

    async def decompose(self) -> List[Knowledge]:
        """
        分解仓库内的知识单元

        Returns:
            List[Knowledge]: 知识列表

        Raises:
            ValueError: 当仓库未正确初始化时
        """
        if not self.local_repo or not self.repo_path:
            raise ValueError("Repository not properly initialized")

        # 统计整个 repo 的总文件大小
        total_size = 0
        for root, _, files in os.walk(self.repo_path):
            if ".git" in root:
                continue
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                except Exception:
                    continue

        # 初始化文件模式管理器
        split_config = getattr(self.knowledge, "split_config", None)
        if split_config and getattr(split_config, "type", None) == "github_repo":
            pattern_manager = FilePatternManager(
                config=split_config, repo_path=self.repo_path
            )
            warnings = pattern_manager.validate_patterns()
            if warnings:
                logger.warning(
                    "Pattern configuration warnings:\n" + "\n".join(warnings)
                )
        else:
            # 创建一个兼容的配置字典
            dummy_config = {
                "include_patterns": ["*.md", "*.mdx"],
                "ignore_patterns": [],
                "no_gitignore": True,
                "no_default_ignore_patterns": False,
            }
            pattern_manager = FilePatternManager(
                config=dummy_config, repo_path=self.repo_path
            )

        current_commit = self.local_repo.head.commit

        github_repo_list: List[Knowledge] = []

        for root, _, files in os.walk(self.repo_path):
            if ".git" in root:
                continue

            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.repo_path)

                if not pattern_manager.should_include_file(relative_path):
                    continue

                try:
                    file_size = os.path.getsize(file_path)
                    git_file_path = relative_path.replace("\\", "/")
                    blob = current_commit.tree / git_file_path
                    file_sha = blob.hexsha
                    ext = os.path.splitext(relative_path)[1].lower()
                    knowledge_type = self.get_knowledge_type_by_ext(ext)
                    file_url = f"{self.base_url}/{self.repo_name}/blob/{self.branch_name}/{relative_path}"
                    knowledge = Knowledge(
                        source_type=KnowledgeSourceEnum.GITHUB_FILE,
                        knowledge_type=knowledge_type,
                        knowledge_name=f"{self.repo_name}/{relative_path}",
                        embedding_model_name=self.knowledge.embedding_model_name,
                        source_config={
                            **self.knowledge.source_config.model_dump(),
                            "path": relative_path,
                        },
                        tenant_id=self.knowledge.tenant_id,
                        file_size=file_size,
                        file_sha=file_sha,
                        space_id=self.knowledge.space_id,
                        split_config=self.knowledge.split_config,
                        parent_id=self.knowledge.knowledge_id,
                        enabled=True,
                        metadata={"_reference_url": file_url},
                    )
                    github_repo_list.append(knowledge)

                except Exception as e:
                    logger.warning(f"Error processing file {relative_path}: {e}")
                    continue

        return github_repo_list

    async def load(self) -> List[Text]:
        """
        返回项目的文件目录树结构信息和作者信息，便于大模型理解
        """
        if not self.repo_path:
            raise ValueError("Repository not properly initialized")

        def build_tree(path: str, prefix: str = "") -> str:
            entries = sorted(os.listdir(path))
            tree_lines = []
            for idx, entry in enumerate(entries):
                full_path = os.path.join(path, entry)
                connector = "└── " if idx == len(entries) - 1 else "├── "
                tree_lines.append(f"{prefix}{connector}{entry}")
                if os.path.isdir(full_path) and entry != ".git":
                    extension = "    " if idx == len(entries) - 1 else "│   "
                    tree_lines.append(build_tree(full_path, prefix + extension))
            return "\n".join(tree_lines)

        root_name = os.path.basename(self.repo_path.rstrip(os.sep))
        tree_str = f"{root_name}\n" + build_tree(self.repo_path)

        try:
            if not self.local_repo:
                raise ValueError("Repository not initialized")

            first_commit = next(
                self.local_repo.iter_commits(
                    rev=self.branch_name, max_count=1, reverse=True
                )
            )
            author_name = first_commit.author.name
            author_email = first_commit.author.email
        except Exception:
            author_name = None
            author_email = None

        return [
            Text(
                content=tree_str,
                metadata={
                    "repo_name": self.repo_name,
                    "author_name": author_name,
                    "author_email": author_email,
                },
            )
        ]

    async def on_load_finished(self) -> None:
        """清理临时资源"""
        try:
            if self.repo_path and os.path.exists(self.repo_path):
                shutil.rmtree(self.repo_path)
                self.repo_path = None
                self.local_repo = None
                logger.info(f"Cleaned up temporary directory for {self.repo_name}")
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")

    async def get_file_by_path(self, path: str) -> GitFileElementType:
        if not self.local_repo:
            raise ValueError("Repository not initialized")

        if not self.repo_path:
            raise ValueError("Repository path not initialized")

        full_path = os.path.join(self.repo_path, path)
        if not os.path.exists(full_path):
            raise ValueError(f"File not found: {path}")

        try:
            blob = self.local_repo.head.commit.tree[path]
            # 类型检查确保source_config是GithubRepoSourceConfig
            if not isinstance(self.knowledge.source_config, GithubRepoSourceConfig):
                raise ValueError("Invalid source config type")
            base_url = self.knowledge.source_config.url.rstrip("/")
            file_url = f"{base_url}/{self.repo_name}/blob/{self.branch_name}/{path}"

            # 读取文件内容
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # 如果不是文本文件，尝试二进制读取并转换为base64
                with open(full_path, "rb") as f:
                    import base64

                    content = base64.b64encode(f.read()).decode("utf-8")

            return GitFileElementType(
                content=content,
                path=path,
                mode=blob.mode_str,
                url=file_url,
                branch=self.branch_name or "main",
                repo_name=self.repo_name,
                size=blob.size,
                sha=blob.hexsha,
            )
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            raise ValueError(f"Failed to get file info for {path}: {str(e)}")
