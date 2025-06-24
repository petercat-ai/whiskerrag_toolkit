import json
from typing import AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from langchain_core.messages import HumanMessage

from whiskerrag_client.agent import AgentClient
from whiskerrag_client.http_client import HttpClient
from whiskerrag_types.model.agent import KnowledgeScope, ProResearchRequest
from whiskerrag_types.model.retrieval import RetrievalConfig


@pytest.fixture
def http_client():
    """创建测试用的 HTTP 客户端"""
    return HttpClient(
        base_url="http://test.example.com", token="test-token", timeout=10
    )


@pytest.fixture
def agent_client(http_client):
    """创建测试用的 Agent 客户端"""
    return AgentClient(http_client)


@pytest.fixture
def sample_request():
    """创建示例请求"""
    return ProResearchRequest(
        messages=[HumanMessage(content="研究人工智能的发展历史")],
        model="test-model",
        number_of_initial_queries=3,
        max_research_loops=2,
        enable_knowledge_retrieval=True,
        enable_web_search=False,
    )


class TestAgentClient:
    """测试 AgentClient 类"""

    def test_initialization(self, http_client):
        """测试 AgentClient 初始化"""
        client = AgentClient(http_client)
        assert client.http_client == http_client
        assert client.base_path == "/v1/api/agent"

        # 测试自定义 base_path
        custom_client = AgentClient(http_client, base_path="/custom/agent")
        assert custom_client.base_path == "/custom/agent"

    @pytest.mark.asyncio
    async def test_pro_research_streaming(self, agent_client, sample_request):
        """测试流式研究方法"""
        # 模拟流式响应数据
        mock_lines = [
            'data: {"type": "start", "content": "开始研究..."}',
            'data: {"type": "progress", "content": "正在搜索相关信息..."}',
            'data: {"type": "result", "content": "人工智能发展历史包括..."}',
            "data: [DONE]",
        ]

        # 创建模拟响应对象
        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        async def mock_aiter_lines():
            for line in mock_lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines

        # 模拟 httpx.AsyncClient.stream 上下文管理器
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None

        with patch.object(
            agent_client.http_client.client, "stream", return_value=mock_stream_context
        ):
            results = []
            async for chunk in agent_client.pro_research(sample_request):
                results.append(chunk)

            # 验证结果
            assert len(results) == 3  # 排除 [DONE] 标记
            assert results[0]["type"] == "start"
            assert results[1]["type"] == "progress"
            assert results[2]["type"] == "result"

    @pytest.mark.asyncio
    async def test_pro_research_json_lines(self, agent_client, sample_request):
        """测试处理 JSON 行格式的响应"""
        mock_lines = [
            '{"step": 1, "action": "query_generation"}',
            '{"step": 2, "action": "knowledge_retrieval"}',
            '{"step": 3, "action": "response_synthesis"}',
        ]

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        async def mock_aiter_lines():
            for line in mock_lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None

        with patch.object(
            agent_client.http_client.client, "stream", return_value=mock_stream_context
        ):
            results = []
            async for chunk in agent_client.pro_research(sample_request):
                results.append(chunk)

            assert len(results) == 3
            assert all("step" in result for result in results)
            assert all("action" in result for result in results)

    @pytest.mark.asyncio
    async def test_pro_research_invalid_json(self, agent_client, sample_request):
        """测试处理无效 JSON 的情况"""
        mock_lines = [
            "data: invalid json content",
            "plain text line",
            'data: {"valid": "json"}',
        ]

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        async def mock_aiter_lines():
            for line in mock_lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None

        with patch.object(
            agent_client.http_client.client, "stream", return_value=mock_stream_context
        ):
            results = []
            async for chunk in agent_client.pro_research(sample_request):
                results.append(chunk)

            assert len(results) == 3
            # 第一个和第二个应该是文本格式
            assert results[0]["type"] == "text"
            assert results[1]["type"] == "text"
            # 第三个应该是有效的 JSON
            assert results[2]["valid"] == "json"

    @pytest.mark.asyncio
    async def test_pro_research_http_error(self, agent_client, sample_request):
        """测试 HTTP 错误处理"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPError("HTTP Error")

        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None

        with patch.object(
            agent_client.http_client.client, "stream", return_value=mock_stream_context
        ):
            with pytest.raises(httpx.HTTPError):
                async for chunk in agent_client.pro_research(sample_request):
                    pass

    @pytest.mark.asyncio
    async def test_pro_research_sync_single_chunk(self, agent_client, sample_request):
        """测试同步方法 - 单个响应块"""
        mock_lines = ['data: {"result": "完整的研究结果", "status": "completed"}']

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        async def mock_aiter_lines():
            for line in mock_lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None

        with patch.object(
            agent_client.http_client.client, "stream", return_value=mock_stream_context
        ):
            result = await agent_client.pro_research_sync(sample_request)

            assert result["result"] == "完整的研究结果"
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_pro_research_sync_multiple_chunks(
        self, agent_client, sample_request
    ):
        """测试同步方法 - 多个响应块"""
        mock_lines = [
            'data: {"content": "第一部分"}',
            'data: {"content": "第二部分"}',
            'data: {"content": "第三部分"}',
        ]

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        async def mock_aiter_lines():
            for line in mock_lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None

        with patch.object(
            agent_client.http_client.client, "stream", return_value=mock_stream_context
        ):
            result = await agent_client.pro_research_sync(sample_request)

            assert result["content"] == "第一部分\n第二部分\n第三部分"
            assert result["type"] == "combined"
            assert result["chunks_count"] == 3

    @pytest.mark.asyncio
    async def test_pro_research_sync_empty_response(self, agent_client, sample_request):
        """测试同步方法 - 空响应"""
        mock_lines = []

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        async def mock_aiter_lines():
            for line in mock_lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None

        with patch.object(
            agent_client.http_client.client, "stream", return_value=mock_stream_context
        ):
            result = await agent_client.pro_research_sync(sample_request)

            assert result["content"] == ""
            assert result["type"] == "empty"

    def test_invalid_http_client(self, sample_request):
        """测试无效的 HTTP 客户端"""
        # 创建一个没有 httpx.AsyncClient 的模拟客户端
        mock_client = Mock()
        mock_client.base_url = "http://test.com"
        mock_client.base_path = "/api"
        mock_client.headers = {}
        mock_client.timeout = 10
        # 不设置 client 属性，或者设置为非 AsyncClient 类型

        agent_client = AgentClient(mock_client)

        # 这应该引发 ValueError
        async def test_invalid_client():
            async for chunk in agent_client.pro_research(sample_request):
                pass

        with pytest.raises(ValueError, match="HTTP client does not support streaming"):
            import asyncio

            asyncio.run(test_invalid_client())

    @pytest.mark.asyncio
    async def test_request_with_complex_parameters(self, agent_client):
        """测试包含复杂参数的请求"""
        scope = KnowledgeScope(
            space_ids=["space1", "space2"],
            tenant_id="test-tenant",
            auth_info="bearer token",
        )
        config = RetrievalConfig(type="semantic_search")

        request = ProResearchRequest(
            messages=[HumanMessage(content="复杂研究请求")],
            model="advanced-model",
            number_of_initial_queries=5,
            max_research_loops=3,
            enable_knowledge_retrieval=True,
            enable_web_search=True,
            knowledge_scope_list=[scope],
            knowledge_retrieval_config=config,
        )

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        async def mock_aiter_lines():
            yield 'data: {"status": "processing", "message": "处理复杂请求"}'

        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None

        with patch.object(
            agent_client.http_client.client, "stream", return_value=mock_stream_context
        ) as mock_stream:
            results = []
            async for chunk in agent_client.pro_research(request):
                results.append(chunk)

            # 验证调用参数
            mock_stream.assert_called_once()
            call_kwargs = mock_stream.call_args.kwargs

            assert call_kwargs["method"] == "POST"
            assert (
                call_kwargs["url"]
                == "http://test.example.com/v1/api/agent/pro_research"
            )
            assert "json" in call_kwargs

            # 验证请求数据包含所有字段
            json_data = call_kwargs["json"]
            assert json_data["model"] == "advanced-model"
            assert json_data["number_of_initial_queries"] == 5
            assert json_data["enable_web_search"] is True
            assert len(json_data["knowledge_scope_list"]) == 1
