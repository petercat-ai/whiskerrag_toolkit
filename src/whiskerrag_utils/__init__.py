from typing import List

from whiskerrag_types.model.chunk import Chunk
from whiskerrag_types.model.knowledge import Knowledge
from whiskerrag_types.model.multi_modal import Image

from .registry import (
    RegisterTypeEnum,
    RetrievalEnum,
    get_register,
    init_register,
    register,
)


async def get_chunks_by_knowledge(knowledge: Knowledge) -> List[Chunk]:
    LoaderCls = get_register(RegisterTypeEnum.KNOWLEDGE_LOADER, knowledge.source_type)
    SplitterCls = get_register(RegisterTypeEnum.SPLITTER, knowledge.knowledge_type)
    EmbeddingCls = get_register(
        RegisterTypeEnum.EMBEDDING, knowledge.embedding_model_name
    )
    content = await LoaderCls(knowledge).load()
    split_result = SplitterCls().split(content, knowledge.split_config)
    chunks = []
    for split_item in split_result:
        if isinstance(split_item, str):
            embedding = await EmbeddingCls().embed_text(split_item, timeout=30)
        elif isinstance(split_item, Image):
            embedding = await EmbeddingCls().embed_image(split_item, timeout=30)
        else:
            print(f"[warn]: illegal split item :{split_item}")
            continue
        chunk = Chunk(
            context=split_item if isinstance(split_item, str) else str(split_item),
            metadata={},
            embedding=embedding,
            knowledge_id=knowledge.knowledge_id,
            embedding_model_name=knowledge.embedding_model_name,
            space_id=knowledge.space_id,
            tenant_id=knowledge.tenant_id,
        )
        chunks.append(chunk)
    return chunks


__all__ = [
    "get_register",
    "register",
    "RegisterTypeEnum",
    "init_register",
    "RetrievalEnum",
    "SplitterEnum",
    "get_chunks_by_knowledge",
]
