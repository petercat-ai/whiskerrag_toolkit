import base64
from typing import List, Optional
from github import Github
from langchain_core.documents import Document

from whiskerrag_types.interface.loader_interface import BaseLoader
from whiskerrag_types.model.knowledge import Knowledge, KnowledgeSourceType
from whiskerrag_utils.registry import register


@register(KnowledgeSourceType.GITHUB_REPO)
class GithubFileLoader(BaseLoader):
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
        if not all(
            key in knowledge.metadata for key in ["repo_name", "path", "branch"]
        ):
            raise ValueError("metadata must contain 'repo_name', 'path', and 'branch'")
        if not knowledge.auth_info:
            raise ValueError("auth_info is required")
        self.github = Github(knowledge.auth_info)
        self.repo_name = knowledge.metadata.get("repo_name")
        self.repo = self.github.get_repo(self.repo_name)
        self.path = knowledge.metadata.get("path")
        self.branch = knowledge.metadata.get("branch")
        self.commit_id = knowledge.metadata.get("commit_id")

    def _get_commit_id_by_branch(self, branch: str) -> str:
        branch_info = self.github.get_branch(branch)
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

    async def load(self) -> List[Document]:
        content = self._get_file_content_by_path()
        metadata = {
            **self.knowledge.metadata,
            "sha": self.sha,
            "size": self.size,
        }
        return [Document(page_content=content, metadata=metadata)]
