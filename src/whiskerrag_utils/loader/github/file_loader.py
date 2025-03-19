import base64
from typing import Optional

from github import Github

from whiskerrag_types.interface.loader_interface import BaseLoader
from whiskerrag_types.model.knowledge import (
    GithubFileSourceConfig,
    Knowledge,
    KnowledgeSourceEnum,
)
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.KNOWLEDGE_LOADER, KnowledgeSourceEnum.GITHUB_FILE)
class GithubFileLoader(BaseLoader):
    """
    Load a file from a GitHub repository.
    """

    knowledge: Knowledge
    path: str
    mode: str
    url: str
    branch: str
    repo_name: str
    size: int
    sha: str
    commit_id: Optional[str]
    github: Github

    def __init__(
        self,
        knowledge: Knowledge,
    ):
        self.knowledge = knowledge
        if not isinstance(knowledge.source_config, GithubFileSourceConfig):
            raise ValueError("source_config should be GithubFileSourceConfig")
        source_config: GithubFileSourceConfig = knowledge.source_config
        self.github = (
            Github(source_config.auth_info) if source_config.auth_info else Github()
        )
        self.repo = self.github.get_repo(source_config.repo_name)
        self.branch = source_config.branch or self.repo.default_branch
        self.commit_id = source_config.commit_id or self._get_commit_id_by_branch(
            self.branch
        )
        self.path = source_config.path

    def _get_commit_id_by_branch(self, branch: str) -> str:
        branch_info = self.repo.get_branch(branch)
        return branch_info.commit.sha

    def _get_file_content_by_path(
        self,
    ) -> str:
        file_content = (
            self.repo.get_contents(self.path, ref=self.commit_id)
            if self.commit_id
            else self.repo.get_contents(self.path)
        )
        if isinstance(file_content, list):
            print("[warn]file_content is a list")
            file_content = file_content[0]
        self.sha = file_content.sha
        self.size = file_content.size
        return base64.b64decode(file_content.content).decode("utf-8")

    async def load(self) -> str:
        content = self._get_file_content_by_path()
        return content
