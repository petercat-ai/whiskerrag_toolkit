# Agent API 客户端

本文档介绍了为 WhiskerRAG Toolkit 创建的 Agent API 客户端，包括类型定义、客户端实现和使用示例。

## 概述

Agent API 客户端提供了与专业研究 (Pro Research) 服务的集成，支持：
- 流式响应处理
- 同步和异步调用
- 复杂参数配置
- 完整的类型安全

## 核心组件

### 1. 类型定义 (`src/whiskerrag_types/model/agent.py`)

#### KnowledgeScope
定义知识检索的范围和权限：

```python
class KnowledgeScope(BaseModel):
    space_ids: Optional[List[str]] = None  # 空间ID列表
    auth_info: str                         # 认证信息
```

#### ProResearchRequest
专业研究请求的完整参数：

```python
class ProResearchRequest(BaseModel):
    messages: List[BaseMessage] = []                    # 对话消息
    model: str = "wohu_qwen3_235b_a22b"                # 使用的模型
    number_of_initial_queries: int = 3                 # 初始查询数量
    max_research_loops: int = 2                        # 最大研究循环数
    enable_knowledge_retrieval: bool = True            # 启用知识检索
    enable_web_search: bool = False                    # 启用网络搜索
    knowledge_scope_list: List[KnowledgeScope] = []    # 知识范围列表
    knowledge_retrieval_config: Optional[RetrievalConfig] = None  # 检索配置
```

### 2. 客户端实现 (`src/whiskerrag_client/agent.py`)

#### AgentClient
提供与 Agent API 交互的方法：

```python
class AgentClient:
    def __init__(self, http_client: BaseClient, base_path: str = "/v1/api/agent")

    async def pro_research(self, request: ProResearchRequest) -> AsyncIterator[Dict[str, Any]]
    async def pro_research_sync(self, request: ProResearchRequest) -> Dict[str, Any]
```

## 使用方法

### 基础用法

```python
from whiskerrag_client.http_client import HttpClient
from whiskerrag_client.agent import AgentClient
from whiskerrag_types.model.agent import ProResearchRequest
from langchain_core.messages import HumanMessage

# 创建客户端
http_client = HttpClient(
    base_url="https://your-api-domain.com",
    token="your-bearer-token"
)
agent_client = AgentClient(http_client)

# 创建请求
request = ProResearchRequest(
    messages=[HumanMessage(content="研究人工智能的发展历史")],
    enable_knowledge_retrieval=True
)

# 流式调用
async for chunk in agent_client.pro_research(request):
    print(chunk)

# 同步调用
result = await agent_client.pro_research_sync(request)
print(result)
```

### 高级配置

```python
from whiskerrag_types.model.agent import KnowledgeScope
from whiskerrag_types.model.retrieval import RetrievalConfig

# 配置知识范围
scope = KnowledgeScope(
    space_ids=["medical-space", "ai-space"],
    tenant_id="your-tenant-id",
    auth_info="bearer your-token"
)

# 配置检索参数
retrieval_config = RetrievalConfig(type="semantic_search")

# 创建高级请求
advanced_request = ProResearchRequest(
    messages=[HumanMessage(content="医疗AI的最新进展")],
    model="gpt-4-turbo",
    number_of_initial_queries=5,
    max_research_loops=3,
    enable_knowledge_retrieval=True,
    enable_web_search=True,
    knowledge_scope_list=[scope],
    knowledge_retrieval_config=retrieval_config
)
```

## 流式响应处理

客户端支持多种流式响应格式：

### Server-Sent Events (SSE)
```
data: {"type": "start", "content": "开始研究..."}
data: {"type": "progress", "content": "正在搜索..."}
data: {"type": "result", "content": "研究结果..."}
data: [DONE]
```

### JSON Lines
```
{"step": 1, "action": "query_generation"}
{"step": 2, "action": "knowledge_retrieval"}
{"step": 3, "action": "response_synthesis"}
```

### 纯文本
```
研究开始...
正在处理查询...
生成最终结果...
```

## 错误处理

```python
try:
    async for chunk in agent_client.pro_research(request):
        # 处理正常响应
        if chunk.get("type") == "error":
            print(f"服务器错误: {chunk.get('message')}")
        else:
            print(f"正常响应: {chunk}")

except httpx.HTTPError as e:
    print(f"HTTP 错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")
```

## 测试

### 类型测试 (`tests/whiskerrag_types/test_agent.py`)
- KnowledgeScope 验证测试
- ProResearchRequest 参数测试
- 序列化/反序列化测试

### 客户端测试 (`tests/whiskerrag_client/test_agent_client.py`)
- 流式响应测试
- 同步调用测试
- 错误处理测试
- 复杂参数测试

运行测试：
```bash
python -m pytest tests/whiskerrag_types/test_agent.py -v
python -m pytest tests/whiskerrag_client/test_agent_client.py -v
```

## API 路由对应

客户端对应的服务器端路由：

```python
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(
    prefix="/v1/api/agent",
    tags=["agent"],
    responses={404: {"description": "Not found"}},
)

@router.post("/pro_research")
async def pro_research(body: ProResearchRequest):
    db_engine = PluginManager().dbPlugin
    return StreamingResponse(db_engine.agent_invoke(body))
```

## 示例文件

完整的使用示例请参考 `examples/agent_client_example.py`，包含：
- 基础和高级用法
- 流式和同步调用
- 错误处理示例
- 资源清理

## 注意事项

1. **认证**: 确保提供有效的 Bearer Token
2. **网络**: 流式调用可能需要较长的超时时间
3. **资源**: 记得关闭 HTTP 客户端以释放资源
4. **错误处理**: 在生产环境中添加适当的重试和错误恢复机制

## 贡献

如需添加新功能或修复问题，请：
1. 添加相应的类型定义
2. 实现客户端方法
3. 编写完整的测试
4. 更新文档
