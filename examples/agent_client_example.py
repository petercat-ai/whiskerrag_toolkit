#!/usr/bin/env python3
"""
AgentClient example

This example shows how to use AgentClient for research tasks,
including streaming response and synchronous call.
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from whiskerrag_client.agent import AgentClient
from whiskerrag_client.http_client import HttpClient
from whiskerrag_types.model.agent import KnowledgeScope, ProResearchRequest
from whiskerrag_types.model.retrieval import RetrievalConfig


async def main():
    """Main function: demonstrate the usage of AgentClient"""

    # 1. create HTTP client
    print("1. create HTTP client...")
    http_client = HttpClient(
        base_url="https://your-api-domain.com",  # replace with actual API address
        token="your-bearer-token",  # replace with actual authentication token
        timeout=30,
    )

    # 2. create Agent client
    print("2. create Agent client...")
    agent_client = AgentClient(http_client)

    # 3. basic research request example
    print("\n=== 基础研究请求示例 ===")
    basic_request = ProResearchRequest(
        messages=[
            HumanMessage(
                content="research the current status and development trend of artificial intelligence in the medical field"
            )
        ],
        model="gpt-4",
        number_of_initial_queries=3,
        max_research_loops=2,
        enable_knowledge_retrieval=True,
        enable_web_search=False,
    )

    print("basic request configuration:")
    print(f"  model: {basic_request.model}")
    print(f"  number of initial queries: {basic_request.number_of_initial_queries}")
    print(f"  maximum research loops: {basic_request.max_research_loops}")
    print(f"  enable knowledge retrieval: {basic_request.enable_knowledge_retrieval}")
    print(f"  enable web search: {basic_request.enable_web_search}")

    print("\n=== streaming call example ===")
    print("start streaming research...")
    try:
        chunk_count = 0
        async for chunk in agent_client.pro_research(basic_request):
            chunk_count += 1
            print(f"received data block {chunk_count}: {chunk}")

            # handle different types of response
            if isinstance(chunk, dict):
                if chunk.get("type") == "start":
                    print("  -> research started")
                elif chunk.get("type") == "progress":
                    print(f"  -> progress updated: {chunk.get('content', '')}")
                elif chunk.get("type") == "result":
                    print(f"  -> research result: {chunk.get('content', '')}")
                elif chunk.get("type") == "error":
                    print(f"  -> error: {chunk.get('message', '')}")
    except Exception as e:
        print(f"streaming call error: {e}")

    # 5. synchronous call example
    print("\n=== synchronous call example ===")
    print("start synchronous research...")
    try:
        result = await agent_client.pro_research_sync(basic_request)
        print("synchronous research completed:")
        print(f"  result type: {result.get('type', 'unknown')}")
        print(f"  content length: {len(str(result.get('content', '')))}")
        if result.get("chunks_count"):
            print(f"  data block count: {result['chunks_count']}")
    except Exception as e:
        print(f"synchronous call error: {e}")

    # 6. complex parameter request example
    print("\n=== complex parameter request example ===")

    # create knowledge scope
    knowledge_scope = KnowledgeScope(
        space_ids=["medical-ai-space", "healthcare-space"],
        tenant_id="your-tenant-id",
        auth_info="bearer your-knowledge-token",
    )

    # create retrieval config
    retrieval_config = RetrievalConfig(type="semantic_search")

    # create complex request
    complex_request = ProResearchRequest(
        messages=[
            HumanMessage(content="what are the latest breakthroughs in medical AI?"),
            AIMessage(
                content="I'll help you research the latest breakthroughs in medical AI..."
            ),
            HumanMessage(
                content="please focus on the application of diagnosis and treatment"
            ),
        ],
        model="gpt-4-turbo",
        number_of_initial_queries=5,
        max_research_loops=3,
        enable_knowledge_retrieval=True,
        enable_web_search=True,
        knowledge_scope_list=[knowledge_scope],
        knowledge_retrieval_config=retrieval_config,
    )

    print("复杂请求配置:")
    print(f"  conversation history length: {len(complex_request.messages)}")
    print(f"  knowledge scope count: {len(complex_request.knowledge_scope_list)}")
    print(f"  retrieval config type: {complex_request.knowledge_retrieval_config.type}")
    print(f"  enable web search: {complex_request.enable_web_search}")

    # 7. error handling example
    print("\n=== error handling example ===")
    try:
        # create a request that may cause an error
        error_request = ProResearchRequest(
            messages=[HumanMessage(content="")],  # empty content may cause an error
            max_research_loops=0,  # invalid loop count
        )

        async for chunk in agent_client.pro_research(error_request):
            print(f"unexpected data received: {chunk}")

    except Exception as e:
        print(f"expected error handling: {type(e).__name__}: {e}")

    # 8. close client
    print("\n=== clean up resources ===")
    await http_client.close()
    print("HTTP client closed")


def sync_example():
    """synchronous call example (using asyncio.run)"""
    print("\n=== synchronous call example (using asyncio.run) ===")

    async def simple_research():
        http_client = HttpClient(
            base_url="https://your-api-domain.com", token="your-bearer-token"
        )
        agent_client = AgentClient(http_client)

        request = ProResearchRequest(
            messages=[HumanMessage(content="simple research question")],
            number_of_initial_queries=1,
            max_research_loops=1,
        )

        try:
            result = await agent_client.pro_research_sync(request)
            print(f"research completed: {result}")
        finally:
            await http_client.close()

    # run async function in synchronous environment
    asyncio.run(simple_research())


if __name__ == "__main__":
    print("AgentClient example")
    print("=" * 50)

    # run main example
    asyncio.run(main())

    # run synchronous example
    sync_example()

    print("\nexample completed!")
    print("\nusage:")
    print("1. replace 'your-api-domain.com' with actual API address")
    print("2. replace 'your-bearer-token' with actual authentication token")
    print("3. adjust request parameters as needed")
    print("4. add appropriate error handling in actual use")
