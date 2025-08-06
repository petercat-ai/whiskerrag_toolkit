from whiskerrag_types.model.knowledge import KnowledgeTypeEnum
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils.loader.git_file_loader import get_github_file_content


def test_get_gitlab_file_content_text_success():
    # 请用你实际可访问的 gitlab/alipay 实例、项目、文件
    repo_name = "petercat-ai/whiskerrag_toolkit"
    branch = "main"
    path = "README.md"
    token = None

    obj = get_github_file_content(
        owner_repo=repo_name,
        branch=branch,
        path=path,
        knowledge_type=KnowledgeTypeEnum.TEXT,
        token=token,
    )
    assert isinstance(obj, Text)
    # content 非空
    assert obj.content
    print(obj.content[0:50])
    # 简单检查 Markdown
    assert isinstance(obj.content, str)
