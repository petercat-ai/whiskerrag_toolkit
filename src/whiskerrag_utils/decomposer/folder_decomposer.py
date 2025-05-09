from typing import Dict, List

from langchain_community.document_loaders.yuque import YuqueLoader

from whiskerrag_types.interface import BaseDecomposer
from whiskerrag_types.model.knowledge import (
    GithubRepoSourceConfig,
    Knowledge,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
    YuqueSourceConfig,
)
from whiskerrag_utils.loader.github.repo_loader import (
    GitFileElementType,
    GithubRepoLoader,
)
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.DECOMPOSER, KnowledgeTypeEnum.FOLDER)
class FolderDecomposer(BaseDecomposer):
    async def decompose(self) -> List[Knowledge]:
        results = []
        if self.knowledge.source_type == KnowledgeSourceEnum.GITHUB_REPO:
            results = await self.get_knowledge_list_from_github_repo(self.knowledge)
        if self.knowledge.source_type == KnowledgeSourceEnum.YUQUE:
            results = await self.get_knowledge_list_from_yuque(self.knowledge)
        return results

    async def get_knowledge_list_from_github_repo(
        self,
        knowledge: Knowledge,
    ) -> List[Knowledge]:
        if isinstance(knowledge.source_config, GithubRepoSourceConfig):
            repo_name = knowledge.source_config.repo_name
        else:
            raise TypeError(
                "source_config must be of type GithubRepoSourceConfig to access repo_name"
            )
        auth_info = knowledge.source_config.auth_info
        branch_name = knowledge.source_config.branch
        github_loader = GithubRepoLoader(repo_name, branch_name, auth_info)
        file_list: List[GitFileElementType] = github_loader.get_file_list()
        github_repo_list: List[Knowledge] = []
        for file in file_list:
            if not file.path.endswith((".md", ".mdx")):
                continue
            else:
                child_knowledge = Knowledge(
                    **knowledge.model_dump(
                        exclude={
                            "source_type",
                            "knowledge_type",
                            "knowledge_name",
                            "source_config",
                            "tenant_id",
                            "file_size",
                            "file_sha",
                            "metadata",
                            "parent_id",
                            "enabled",
                        }
                    ),
                    source_type=KnowledgeSourceEnum.GITHUB_FILE,
                    knowledge_type=KnowledgeTypeEnum.MARKDOWN,
                    knowledge_name=f"{file.repo_name}/{file.path}",
                    source_config={
                        **knowledge.source_config.model_dump(),
                        "path": file.path,
                    },
                    tenant_id=knowledge.tenant_id,
                    file_size=file.size,
                    file_sha=file.sha,
                    metadata={},
                    parent_id=knowledge.knowledge_id,
                    enabled=True,
                )
                github_repo_list.append(child_knowledge)
        return github_repo_list

    async def get_knowledge_list_from_yuque(
        self,
        knowledge: Knowledge,
    ) -> List[Knowledge]:
        if not isinstance(knowledge.source_config, YuqueSourceConfig):
            raise TypeError("source_config must be of type YuqueSourceConfig")

        knowledge_list: List[Knowledge] = []
        loader = YuqueLoader(
            access_token=knowledge.source_config.auth_info,
            api_url=knowledge.source_config.api_url,
        )

        def get_docs_by_book_id(book_id: int) -> List[Dict]:
            document_ids = loader.get_document_ids(book_id=book_id)
            documents = []
            for doc_id in document_ids:
                documents.append(
                    loader.get_document(book_id=book_id, document_id=doc_id)
                )
            return documents

        try:
            # Case 1: If document_id is provided, create single knowledge
            if knowledge.source_config.document_id:
                if not knowledge.source_config.book_id:
                    raise ValueError("book_id is required when document_id is provided")
                document = loader.get_document(
                    int(knowledge.source_config.book_id),
                    int(knowledge.source_config.document_id),
                )
                knowledge_list.append(
                    Knowledge(
                        space_id=knowledge.space_id,
                        source_type=KnowledgeSourceEnum.YUQUE,
                        knowledge_type=KnowledgeTypeEnum.YUQUEDOC,
                        knowledge_name=document.get("title", ""),
                        split_config=knowledge.split_config.model_dump(),
                        source_config=YuqueSourceConfig(
                            api_url=knowledge.source_config.api_url,
                            group_id=knowledge.source_config.group_id,
                            book_id=knowledge.source_config.book_id,
                            document_id=knowledge.source_config.document_id,
                            auth_info=knowledge.source_config.auth_info,
                        ),
                        tenant_id=knowledge.tenant_id,
                        file_size=0,  # You might want to calculate actual size
                        file_sha="",  # You might want to calculate actual sha
                        metadata={"document_id": document.get("id")},
                        parent_id=knowledge.knowledge_id,
                        enabled=True,
                    )
                )

            # Case 2: If only book_id is provided, create knowledge for each document in the book
            elif knowledge.source_config.book_id:
                documents = get_docs_by_book_id(int(knowledge.source_config.book_id))
                for doc in documents:
                    doc_id = doc.get("id")
                    if doc_id is None:
                        continue

                    knowledge_list.append(
                        Knowledge(
                            space_id=knowledge.space_id,
                            source_type=KnowledgeSourceEnum.YUQUE,
                            knowledge_type=KnowledgeTypeEnum.YUQUEDOC,
                            knowledge_name=doc.get("title", ""),
                            source_config=YuqueSourceConfig(
                                api_url=knowledge.source_config.api_url,
                                group_id=knowledge.source_config.group_id,
                                book_id=knowledge.source_config.book_id,
                                document_id=doc_id,
                                auth_info=knowledge.source_config.auth_info,
                            ),
                            split_config=knowledge.split_config.model_dump(),
                            tenant_id=knowledge.tenant_id,
                            file_size=0,
                            file_sha="",
                            metadata={"document_id": doc_id},
                            parent_id=knowledge.knowledge_id,
                            enabled=True,
                        )
                    )

            # Case 3: If only group_id is provided, get all books and their documents
            else:
                books = loader.get_books(user_id=loader.get_user_id())
                for book in books:
                    book_id = book.get("id")
                    if book_id is None:
                        raise Exception(
                            f"can not get id from knowledge:{knowledge.source_config}"
                        )
                    documents = documents = get_docs_by_book_id(book_id)
                    for doc in documents:
                        doc_id = doc.get("id")
                        if doc_id is None:
                            continue

                        knowledge_list.append(
                            Knowledge(
                                space_id=knowledge.space_id,
                                source_type=KnowledgeSourceEnum.YUQUE,
                                knowledge_type=KnowledgeTypeEnum.YUQUEDOC,
                                knowledge_name=doc.get("title", ""),
                                split_config=knowledge.split_config.model_dump(),
                                source_config=YuqueSourceConfig(
                                    api_url=knowledge.source_config.api_url,
                                    group_id=knowledge.source_config.group_id,
                                    book_id=book_id,
                                    document_id=doc_id,
                                    auth_info=knowledge.source_config.auth_info,
                                ),
                                tenant_id=knowledge.tenant_id,
                                file_size=0,
                                file_sha="",
                                metadata={"document_id": doc_id},
                                parent_id=knowledge.knowledge_id,
                                enabled=True,
                            )
                        )

            return knowledge_list

        except Exception as e:
            raise Exception(f"Failed to get knowledge list from Yuque: {e}")
