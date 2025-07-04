#!/usr/bin/env python3
"""
Example demonstrating the enhanced JSON parser functionality.
This example shows how the JSON parser now supports both dictionary and array formats.
"""

import asyncio

from whiskerrag_types.model.knowledge import Knowledge, KnowledgeTypeEnum
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils.registry import RegisterTypeEnum, get_register, init_register


async def main():
    """Demonstrate JSON parser with different input formats."""

    # Initialize the registry
    init_register()

    # Get the JSON parser
    JSONParserCls = get_register(RegisterTypeEnum.PARSER, "json")
    parser = JSONParserCls()

    # Test configuration
    split_config = {
        "type": "json",
        "max_chunk_size": 50,
        "min_chunk_size": 10,
    }

    # Example 1: Dictionary format JSON
    dict_json = """
    {
        "user": {
            "name": "John Doe",
            "age": 30,
            "email": "john.doe@example.com",
            "preferences": {
                "theme": "dark",
                "language": "en"
            }
        },
        "settings": {
            "notifications": true,
            "auto_save": false
        }
    }
    """

    knowledge_dict = Knowledge(
        source_type="user_input_text",
        knowledge_type=KnowledgeTypeEnum.JSON,
        space_id="example_space",
        knowledge_name="dict_example",
        split_config=split_config,
        source_config={"text": dict_json},
        embedding_model_name="openai",
        tenant_id="example_tenant",
        metadata={},
    )

    print("=== Testing Dictionary Format JSON ===")
    result_dict = await parser.parse(
        knowledge_dict,
        Text(content=dict_json, metadata={}),
    )

    print(f"Dictionary JSON split into {len(result_dict)} chunks:")
    for i, chunk in enumerate(result_dict):
        print(f"Chunk {i+1}: {chunk.content[:100]}...")

    print("\n" + "="*50 + "\n")

    # Example 2: Array format JSON
    array_json = """
    [
        {
            "id": 1,
            "name": "Product A",
            "price": 29.99,
            "category": "electronics"
        },
        {
            "id": 2,
            "name": "Product B",
            "price": 49.99,
            "category": "clothing"
        },
        {
            "id": 3,
            "name": "Product C",
            "price": 19.99,
            "category": "books"
        }
    ]
    """

    knowledge_array = Knowledge(
        source_type="user_input_text",
        knowledge_type=KnowledgeTypeEnum.JSON,
        space_id="example_space",
        knowledge_name="array_example",
        split_config=split_config,
        source_config={"text": array_json},
        embedding_model_name="openai",
        tenant_id="example_tenant",
        metadata={},
    )

    print("=== Testing Array Format JSON ===")
    result_array = await parser.parse(
        knowledge_array,
        Text(content=array_json, metadata={}),
    )

    print(f"Array JSON split into {len(result_array)} chunks:")
    for i, chunk in enumerate(result_array):
        print(f"Chunk {i+1}: {chunk.content[:100]}...")

    print("\n" + "="*50 + "\n")

    # Example 3: Error handling
    print("=== Testing Error Handling ===")

    # Invalid JSON
    invalid_json = '{"invalid": json}'

    try:
        await parser.parse(
            knowledge_dict,
            Text(content=invalid_json, metadata={}),
        )
    except ValueError as e:
        print(f"Expected error for invalid JSON: {e}")

    # Non-dict/non-array JSON
    primitive_json = "42"

    try:
        await parser.parse(
            knowledge_dict,
            Text(content=primitive_json, metadata={}),
        )
    except ValueError as e:
        print(f"Expected error for primitive JSON: {e}")


if __name__ == "__main__":
    asyncio.run(main())
