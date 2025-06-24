import pytest
from pydantic import ValidationError

from whiskerrag_types.model.agent import KnowledgeScope, ProResearchRequest
from whiskerrag_types.model.retrieval import RetrievalConfig


class TestKnowledgeScope:
    """测试 KnowledgeScope 类型"""

    def test_valid_knowledge_scope(self) -> None:
        """测试有效的 KnowledgeScope 创建"""
        scope = KnowledgeScope(
            space_ids=["space1", "space2"],
            tenant_id="test-tenant-123",
            auth_info="bearer token123",
        )
        assert scope.space_ids == ["space1", "space2"]
        assert scope.tenant_id == "test-tenant-123"
        assert scope.auth_info == "bearer token123"

    def test_knowledge_scope_without_space_ids(self) -> None:
        """测试没有 space_ids 的 KnowledgeScope"""
        scope = KnowledgeScope(tenant_id="test-tenant-123", auth_info="bearer token123")
        assert scope.space_ids is None
        assert scope.tenant_id == "test-tenant-123"
        assert scope.auth_info == "bearer token123"

    def test_knowledge_scope_empty_space_ids(self) -> None:
        """测试空的 space_ids 列表"""
        scope = KnowledgeScope(
            space_ids=[], tenant_id="test-tenant-123", auth_info="bearer token123"
        )
        assert scope.space_ids == []
        assert scope.tenant_id == "test-tenant-123"

    def test_knowledge_scope_required_fields(self) -> None:
        """测试必需字段的验证"""
        # 缺少 tenant_id
        with pytest.raises(ValidationError):
            KnowledgeScope(auth_info="bearer token123")

        # 缺少 auth_info
        with pytest.raises(ValidationError):
            KnowledgeScope(tenant_id="test-tenant-123")

    def test_knowledge_scope_serialization(self) -> None:
        """测试序列化"""
        scope = KnowledgeScope(
            space_ids=["space1", "space2"],
            tenant_id="test-tenant-123",
            auth_info="bearer token123",
        )
        data = scope.model_dump()
        assert data == {
            "space_ids": ["space1", "space2"],
            "tenant_id": "test-tenant-123",
            "auth_info": "bearer token123",
        }


class TestProResearchRequest:
    """测试 ProResearchRequest 类型"""

    def test_default_values(self) -> None:
        """测试默认值"""
        request = ProResearchRequest()
        assert request.messages == []
        assert request.model == "wohu_qwen3_235b_a22b"
        assert request.number_of_initial_queries == 3
        assert request.max_research_loops == 2
        assert request.enable_knowledge_retrieval is True
        assert request.enable_web_search is False
        assert request.knowledge_scope_list == []
        assert request.knowledge_retrieval_config is None

    def test_with_messages(self) -> None:
        """测试包含消息的请求"""
        messages = [
            {"content": "Hello", "role": "user"},
            {"content": "Hi there!", "role": "assistant"},
        ]
        request = ProResearchRequest(messages=messages)
        assert len(request.messages) == 2
        assert request.messages[0].content == "Hello"
        assert request.messages[1].content == "Hi there!"

    def test_custom_model(self) -> None:
        """测试自定义模型"""
        request = ProResearchRequest(model="custom-model-v1")
        assert request.model == "custom-model-v1"

    def test_custom_query_count(self) -> None:
        """测试自定义查询数量"""
        request = ProResearchRequest(number_of_initial_queries=5)
        assert request.number_of_initial_queries == 5

    def test_custom_research_loops(self) -> None:
        """测试自定义研究循环次数"""
        request = ProResearchRequest(max_research_loops=5)
        assert request.max_research_loops == 5

    def test_disable_knowledge_retrieval(self) -> None:
        """测试禁用知识检索"""
        request = ProResearchRequest(enable_knowledge_retrieval=False)
        assert request.enable_knowledge_retrieval is False

    def test_enable_web_search(self) -> None:
        """测试启用网络搜索"""
        request = ProResearchRequest(enable_web_search=True)
        assert request.enable_web_search is True

    def test_with_knowledge_scope(self) -> None:
        """测试包含知识范围的请求"""
        scope1 = KnowledgeScope(
            space_ids=["space1"], tenant_id="tenant1", auth_info="auth1"
        )
        scope2 = KnowledgeScope(tenant_id="tenant2", auth_info="auth2")

        request = ProResearchRequest(knowledge_scope_list=[scope1, scope2])
        assert len(request.knowledge_scope_list) == 2
        assert request.knowledge_scope_list[0].space_ids == ["space1"]
        assert request.knowledge_scope_list[1].space_ids is None

    def test_with_retrieval_config(self) -> None:
        """测试包含检索配置的请求"""
        config = RetrievalConfig(type="deep_retrieval")
        request = ProResearchRequest(knowledge_retrieval_config=config)
        assert request.knowledge_retrieval_config is not None
        assert request.knowledge_retrieval_config.type == "deep_retrieval"

    def test_complete_request(self) -> None:
        """测试完整的请求配置"""
        messages = [{"content": "Research about AI", "role": "user"}]
        scope = KnowledgeScope(
            space_ids=["ai-knowledge"],
            tenant_id="research-tenant",
            auth_info="bearer abc123",
        )
        config = RetrievalConfig(type="semantic_search")

        request = ProResearchRequest(
            messages=messages,
            model="custom-research-model",
            number_of_initial_queries=5,
            max_research_loops=3,
            enable_knowledge_retrieval=True,
            enable_web_search=True,
            knowledge_scope_list=[scope],
            knowledge_retrieval_config=config,
        )

        assert len(request.messages) == 1
        assert request.model == "custom-research-model"
        assert request.number_of_initial_queries == 5
        assert request.max_research_loops == 3
        assert request.enable_knowledge_retrieval is True
        assert request.enable_web_search is True
        assert len(request.knowledge_scope_list) == 1
        assert request.knowledge_retrieval_config.type == "semantic_search"

    def test_serialization(self) -> None:
        """测试序列化"""
        scope = KnowledgeScope(
            space_ids=["space1"], tenant_id="tenant1", auth_info="auth1"
        )
        config = RetrievalConfig(type="test_retrieval")

        request = ProResearchRequest(
            model="test-model",
            number_of_initial_queries=2,
            knowledge_scope_list=[scope],
            knowledge_retrieval_config=config,
        )

        data = request.model_dump()
        assert data["model"] == "test-model"
        assert data["number_of_initial_queries"] == 2
        assert len(data["knowledge_scope_list"]) == 1
        assert data["knowledge_retrieval_config"]["type"] == "test_retrieval"

    @pytest.mark.parametrize(
        "query_count,loop_count",
        [
            (1, 1),
            (10, 5),
            (100, 10),
        ],
    )
    def test_different_counts(self, query_count: int, loop_count: int) -> None:
        """测试不同的查询和循环计数"""
        request = ProResearchRequest(
            number_of_initial_queries=query_count, max_research_loops=loop_count
        )
        assert request.number_of_initial_queries == query_count
        assert request.max_research_loops == loop_count

    def test_field_descriptions(self) -> None:
        """测试字段描述是否正确设置"""
        schema = ProResearchRequest.model_json_schema()

        # 检查重要字段的描述
        assert "messages" in schema["properties"]
        assert "model" in schema["properties"]
        assert "number_of_initial_queries" in schema["properties"]
        assert "max_research_loops" in schema["properties"]
        assert "enable_knowledge_retrieval" in schema["properties"]
        assert "enable_web_search" in schema["properties"]
        assert "knowledge_scope_list" in schema["properties"]
        assert "knowledge_retrieval_config" in schema["properties"]

    def test_boolean_flags_combination(self) -> None:
        """测试布尔标志的各种组合"""
        test_cases = [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]

        for knowledge_enabled, web_enabled in test_cases:
            request = ProResearchRequest(
                enable_knowledge_retrieval=knowledge_enabled,
                enable_web_search=web_enabled,
            )
            assert request.enable_knowledge_retrieval == knowledge_enabled
            assert request.enable_web_search == web_enabled
