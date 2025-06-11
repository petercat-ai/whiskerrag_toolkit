#!/usr/bin/env python3
"""测试LLM注册系统"""

import asyncio
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "../../../src")
sys.path.insert(0, src_dir)

from whiskerrag_types.model.knowledge import Knowledge, KnowledgeTypeEnum
from whiskerrag_types.model.knowledge_source import (
    KnowledgeSourceEnum,
    TextSourceConfig,
)
from whiskerrag_types.model.splitter import TextSplitConfig
from whiskerrag_utils.registry import RegisterTypeEnum, get_register, init_register


async def test_llm_registry():
    """测试LLM注册系统"""
    print("=== 测试LLM注册系统 ===")

    # 初始化注册器
    print("\n1. 初始化注册器...")
    init_register("whiskerrag_utils")

    try:
        ExampleLLM = get_register(RegisterTypeEnum.LLM, "example")
        print(f"成功获取LLM: {ExampleLLM.__name__}")
    except KeyError as e:
        print(f"获取LLM失败: {e}")
        return

    # 创建测试Knowledge
    print("\n3. 创建测试Knowledge...")
    test_knowledge = Knowledge(
        space_id="test-space",
        tenant_id="38fbd78b-1869-482c-9142-e43a2c2s6e42",
        knowledge_name="测试知识",
        knowledge_type=KnowledgeTypeEnum.TEXT,
        source_type=KnowledgeSourceEnum.USER_INPUT_TEXT,
        source_config=TextSourceConfig(text="这是一个测试知识"),
        split_config=TextSplitConfig(
            type="text",
            chunk_size=5,
            chunk_overlap=0,
            separators=["\n\n", "\n", " "],
            keep_separator=False,
            is_separator_regex=False,
        ),
    )

    # 实例化LLM
    print("\n4. 实例化LLM...")
    llm = ExampleLLM(test_knowledge)

    # 测试聊天功能
    print("\n5. 测试聊天功能...")
    response = await llm.chat("你好，世界！")
    print(f"LLM响应: {response.content}")
    print(f"模型: {response.model}")
    print(f"使用情况: {response.usage}")

    # 测试流式聊天
    print("\n6. 测试流式聊天...")
    print("流式响应: ", end="")
    async for chunk in llm.stream_chat("这是一个流式测试"):
        print(chunk.content, end="")
    print()  # 换行

    # 测试多模态输入
    print("\n7. 测试多模态输入...")
    mixed_content = [
        "请看这张图片：",
        llm.prepare_image_content(image_url="https://example.com/image.jpg"),
        "这是什么？",
    ]
    response = await llm.chat(mixed_content)
    print(f"多模态响应: {response.content}")

    # 测试同步方法
    print("\n8. 测试同步方法...")
    sync_response = llm.chat_sync("同步测试")
    print(f"同步响应: {sync_response.content}")

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    asyncio.run(test_llm_registry())
