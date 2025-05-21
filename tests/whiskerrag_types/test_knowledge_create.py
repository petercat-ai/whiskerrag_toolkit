from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
)
from whiskerrag_types.model.knowledge_create import (
    ImageCreate,
    JSONCreate,
    MarkdownCreate,
    QACreate,
    TextCreate,
)


class TestKnowledge:
    def test_TextCreate(self) -> None:
        data = {
            "space_id": "test_space",
            "knowledge_type": KnowledgeTypeEnum.TEXT,
            "knowledge_name": "Test Knowledge",
            "source_type": KnowledgeSourceEnum.USER_INPUT_TEXT,
            "source_config": {"text": "This is a test text for knowledge creation."},
            "embedding_model_name": EmbeddingModelEnum.OPENAI,
            "split_config": {
                "type": "text",
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": ["\n\n", "##"],
                "is_separator_regex": True,
            },
        }
        knowledge_create = TextCreate(**data)
        assert knowledge_create.space_id == "test_space"
        assert knowledge_create.knowledge_type == KnowledgeTypeEnum.TEXT
        assert knowledge_create.knowledge_name == "Test Knowledge"
        assert knowledge_create.source_type == KnowledgeSourceEnum.USER_INPUT_TEXT
        assert (
            knowledge_create.source_config.text
            == "This is a test text for knowledge creation."
        )
        assert knowledge_create.embedding_model_name == EmbeddingModelEnum.OPENAI
        assert knowledge_create.split_config.type == "text"
        assert knowledge_create.split_config.chunk_size == 500
        assert knowledge_create.split_config.chunk_overlap == 100
        assert knowledge_create.split_config.separators == ["\n\n", "##"]
        assert knowledge_create.split_config.is_separator_regex is True

    def test_QACreate(self) -> None:
        text = "This is a test text for knowledge creation."
        data = {
            "knowledge_name": "test_qa",
            "space_id": "test_space",
            "source_type": KnowledgeSourceEnum.USER_INPUT_TEXT,
            "knowledge_type": KnowledgeTypeEnum.QA,
            "question": text,
            "answer": "world",
            "split_config": {
                "type": "text",
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": ["\n\n", "##"],
                "is_separator_regex": False,
            },
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
        }
        knowledge = QACreate(**data)
        assert knowledge.source_config.text == text
        assert knowledge.metadata == {"answer": "world"}

    def test_ImageCreate(self) -> None:
        data = {
            "knowledge_name": "test_qa",
            "space_id": "test_space",
            "source_type": KnowledgeSourceEnum.CLOUD_STORAGE_IMAGE,
            "knowledge_type": KnowledgeTypeEnum.IMAGE,
            "source_config": {"url": "112233"},
            "split_config": {"type": "image"},
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
            "file_size": 122,
            "file_sha": "123213",
        }
        knowledge = ImageCreate(**data)
        assert knowledge.knowledge_type == KnowledgeTypeEnum.IMAGE

    def test_MarkdownCreate(self) -> None:
        data = {
            "space_id": "a0da74f3-cb27-44fc-9399-e6a045f85179",
            "embedding_model_name": "bge-base-chinese-1117",
            "knowledge_type": "markdown",
            "source_type": "user_input_text",
            "knowledge_name": "111",
            "source_config": {
                "text": "# 雪鸮 Agent Platform\n\n## 快速启动\n\n```bash\n# 安装依赖\n$ tnpm i\n\n# 本地开发\n# 配置 host 127.0.0.1 local.dev.alipay.net\n$ tnpm run dev\n\n# 联调模式，不走 MOCK\n$ tnpm run devs\n```\n\n## 后端联调\n\n### 本地服务联调\n\n1. 后端启动 xuexiaoagent 本地服务，通常是 http://local.dev.alipay.net:7001\n2. 后端执行 oneapi:upload script,在 OneAPI 创建个人分支 my-branch (名称在 oneapi 中心 https://oneapi.alipay.com/app/xuexiaoagent/tag 可查到，这是只是举例)\n3. 打开前端工程文件 config/config.ts , 修改 XUEXIAO_ONEAPI_TAG = 'my-branch'\n4. 前端工程执行 tnpm run oneapi:service , 发现自动更新了接口等定义\n5. 启动项目后，联调需要进入登陆态，可以通过访问http://local.dev.alipay.net:8000/lui/market 在 network 里找一个登陆的 url 进行登陆。\n\n## 参考文档\n\n- Bigfish 官网：https://bigfish.antfin-inc.com/\n"
            },
            "split_config": {
                "type": "markdown",
                "chunk_size": 1500,
                "chunk_overlap": 150,
                "separators": ["\n\n"],
                "is_separator_regex": False,
            },
        }
        knowledge = MarkdownCreate(**data)
        assert knowledge.knowledge_type == KnowledgeTypeEnum.MARKDOWN

    def test_JSONCreate(self) -> None:
        data = {
            "space_id": "a0da74f3-cb27-44fc-9399-e6a045f85179",
            "knowledge_type": "json",
            "source_type": "user_input_text",
            "knowledge_name": "oneapi",
            "source_config": {"text": "{\n   }"},
            "split_config": {
                "type": "json",
                "max_chunk_size": 2000,
                "min_chunk_size": 20,
            },
            "metadata": {},
        }
        knowledge = JSONCreate(**data)
        assert knowledge.knowledge_type == KnowledgeTypeEnum.JSON
