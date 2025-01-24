from typing import List
from whiskerrag_types.model.knowledge import Knowledge, KnowledgeCreate, ResourceType
from whiskerrag_types.model.task import Task, TaskStatus
from whiskerrag_types.model.tenant import Tenant
from whiskerrag_utils.github.repo_loader import GitFileElementType, GithubRepoLoader
from .sha_util import calculate_sha256


async def gen_knowledge_list(
    request_body: List[KnowledgeCreate], tenant: Tenant
) -> List[Knowledge]:
    knowledge_list: List[Knowledge] = []
    for record in request_body:
        if record.knowledge_type == ResourceType.GITHUB_REPO:
            repo_knowledge_list = await get_knowledge_list_from_github_repo(
                record, tenant
            )
            knowledge_list.extend(repo_knowledge_list)
            continue
        if record.knowledge_type == ResourceType.GITHUB_FILE:
            knowledge = await get_knowledge_from_github_file(record, tenant)
            knowledge_list.append(knowledge)
            continue
        if record.knowledge_type == ResourceType.TEXT:
            knowledge = await get_knowledge_from_text(record, tenant)
            knowledge_list.append(knowledge)
            continue
        print(f"knowledge_type {record.knowledge_type} is not supported")
        continue
    return knowledge_list


async def gen_task_from_knowledge(
    knowledge_list: List[Knowledge], tenant: Tenant
) -> List[Task]:
    task_list: List[Task] = []
    for knowledge in knowledge_list:
        task = Task(
            status=TaskStatus.PENDING,
            knowledge_id=knowledge.knowledge_id,
            space_id=knowledge.tenant_id,
            tenant_id=tenant.tenant_id,
        )
        task_list.append(task)
    return task_list


async def get_knowledge_from_github_file(
    knowledge_create: KnowledgeCreate,
    tenant: Tenant,
) -> Knowledge:
    return Knowledge(
        **knowledge_create.model_dump(),
        tenant_id=tenant.tenant_id,
    )


async def get_knowledge_from_text(
    knowledge_create: KnowledgeCreate,
    tenant: Tenant,
) -> Knowledge:
    return Knowledge(
        **knowledge_create,
        file_sha256=calculate_sha256(knowledge_create.content),
        tenant_id=tenant.tenant_id,
    )


async def get_knowledge_list_from_github_repo(
    knowledge_create: KnowledgeCreate,
    tenant: Tenant,
) -> List[Knowledge]:
    repo_url = knowledge_create.source_url
    repo_name = knowledge_create.knowledge_name
    auth_info = knowledge_create.auth_info
    branch_name = None
    # Get the branch name from knowledge.source_url. If it contains tree/xxx, then xxx is the branch name. For example,
    # in https://github.com/petercat-ai/petercat/tree/fix/sqs-executes, the branch name is fix/sqs-executes.
    if "tree/" in repo_url:
        branch_name = repo_url.split("tree/")[1]
    github_loader = GithubRepoLoader(repo_name, branch_name, auth_info)
    file_list: List[GitFileElementType] = github_loader.get_file_list()
    github_repo_list: List[Knowledge] = []
    for file in file_list:
        if not file.path.endswith(".md"):
            continue
        else:
            knowledge = Knowledge(
                **knowledge_create.model_dump(
                    exclude={
                        "knowledge_type",
                        "knowledge_name",
                        "source_url",
                        "tenant_id",
                        "file_size",
                        "file_sha",
                    }
                ),
                knowledge_type=ResourceType.GITHUB_FILE,
                knowledge_name=f"{file.repo_name}/{file.path}",
                source_url=file.url,
                tenant_id=tenant.tenant_id,
                file_size=file.size,
                file_sha=file.sha,
            )
            github_repo_list.append(knowledge)
    return github_repo_list
