import asyncio
from typing import List, Optional, Union

from whiskerrag_types.interface.embed_interface import BaseEmbedding
from whiskerrag_types.interface.parser_interface import ParseResult
from whiskerrag_types.model.chunk import Chunk
from whiskerrag_types.model.knowledge import Knowledge
from whiskerrag_types.model.multi_modal import Image, Text

from .registry import RegisterTypeEnum, get_register, init_register, register


async def process_parse_item(
    parse_item: Union[Text, Image],
    knowledge: Knowledge,
    EmbeddingCls: type[BaseEmbedding],
    semaphore: asyncio.Semaphore,
) -> Optional[Chunk]:
    """处理单个解析项的异步函数"""
    async with semaphore:
        try:
            if isinstance(parse_item, Text):
                embedding = await EmbeddingCls().embed_text(
                    parse_item.content, timeout=30
                )
            elif isinstance(parse_item, Image):
                embedding = await EmbeddingCls().embed_image(parse_item, timeout=30)
            else:
                print(f"[warn]: illegal split item :{parse_item}")
                return None

            combined_metadata = {**knowledge.metadata}
            if isinstance(parse_item, Text) and parse_item.metadata:
                combined_metadata.update(parse_item.metadata)

            return Chunk(
                context=(
                    parse_item.content
                    if isinstance(parse_item, Text)
                    else parse_item.url
                ),
                enabled=knowledge.enabled,
                metadata=combined_metadata,
                embedding=embedding,
                knowledge_id=knowledge.knowledge_id,
                embedding_model_name=knowledge.embedding_model_name,
                space_id=knowledge.space_id,
                tenant_id=knowledge.tenant_id,
            )
        except Exception as e:
            print(f"Error processing parse item: {e}")
            return None


async def get_chunks_by_knowledge(knowledge: Knowledge) -> List[Chunk]:
    """
    Convert knowledge into vectorized chunks with controlled concurrency

    Args:
        knowledge (Knowledge): Knowledge object containing source type, split configuration,
                             embedding model and other information

    Returns:
        List[Chunk]: List of vectorized chunks

    Process flow:
    1. Get corresponding loader based on knowledge source type
    2. Get text splitter
    3. Get embedding model
    4. Load content as Text or Image
    5. Split content
    6. Vectorize each split content
    7. Generate final list of Chunk objects
    """
    LoaderCls = get_register(RegisterTypeEnum.KNOWLEDGE_LOADER, knowledge.source_type)
    split_type = getattr(knowledge.split_config, "type", "base")

    SplitterCls = get_register(RegisterTypeEnum.PARSER, split_type)
    EmbeddingCls = get_register(
        RegisterTypeEnum.EMBEDDING, knowledge.embedding_model_name
    )

    contents = await LoaderCls(knowledge).load()
    parse_results: ParseResult = []
    for content in contents:
        split_result = await SplitterCls().parse(knowledge, content)
        parse_results.extend(split_result)

    # 创建信号量控制并发
    semaphore = asyncio.Semaphore(3)  # 限制并发数为3

    # 创建所有任务
    tasks = [
        process_parse_item(parse_item, knowledge, EmbeddingCls, semaphore)
        for parse_item in parse_results
    ]

    # 并发执行所有任务
    chunks = await asyncio.gather(*tasks)

    # 过滤掉None值并返回结果
    return [chunk for chunk in chunks if chunk is not None]


__all__ = [
    "get_register",
    "register",
    "RegisterTypeEnum",
    "init_register",
    "SplitterEnum",
    "get_chunks_by_knowledge",
]
