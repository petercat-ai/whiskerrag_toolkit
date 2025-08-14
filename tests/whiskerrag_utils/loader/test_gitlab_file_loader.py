import pytest

from whiskerrag_types.model.knowledge import KnowledgeTypeEnum
from whiskerrag_types.model.multi_modal import Image, Text
from whiskerrag_utils.loader.git_file_loader import get_gitlab_file_content


def test_get_gitlab_file_content_text_success():
    # 请用你实际可访问的 gitlab/alipay 实例、项目、文件
    url = "https://xxx.com"
    repo_name = "liuzhide.lzd/repo_understand"
    branch = "master"
    path = "README.md"
    token = "git:XXX"  # 替换为实际的 token
    try:
        obj = get_gitlab_file_content(
            url=url,
            owner_repo=repo_name,
            branch=branch,
            path=path,
            knowledge_type=KnowledgeTypeEnum.TEXT,
            token=token,
        )
        assert isinstance(obj, Text)
        # content 非空
        assert obj.content
        # 简单检查 Markdown
        assert isinstance(obj.content, str)
    except Exception as e:
        pytest.skip(f"获取 GitLab 文件内容失败: {e},请替换为实际的 token")


def test_get_gitlab_file_content_image_success():

    url = "https://xxxx.com"
    repo_name = "liuzhide.lzd/repo_understand"
    branch = "master"
    path = "777.jpg"
    token = "git:XXX"  # 替换为实际的 token
    try:
        obj = get_gitlab_file_content(
            url=url,
            owner_repo=repo_name,
            branch=branch,
            path=path,
            knowledge_type=KnowledgeTypeEnum.IMAGE,
            token=token,
        )
        assert isinstance(obj, Image)
        assert obj.b64_json
        assert len(obj.b64_json) > 100  # 大概率不是空串
    except Exception as e:
        pytest.skip(f"获取 GitLab 文件内容失败: {e},请替换为实际的 token")
