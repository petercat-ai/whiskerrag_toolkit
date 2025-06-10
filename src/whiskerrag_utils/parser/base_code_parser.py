from typing import List, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter

from whiskerrag_types.interface.parser_interface import BaseParser, ParseResult
from whiskerrag_types.model.knowledge import Knowledge
from whiskerrag_types.model.multi_modal import Image, Text
from whiskerrag_types.model.splitter import BaseCodeSplitConfig
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.PARSER, "base_code")
class CodeParser(BaseParser[Text]):
    async def parse(
        self,
        knowledge: Knowledge,
        content: Text,
    ) -> ParseResult:
        split_config = knowledge.split_config
        if not isinstance(split_config, BaseCodeSplitConfig):
            raise TypeError("knowledge.split_config must be of type CodeSplitConfig")
        splitter = RecursiveCharacterTextSplitter.from_language(
            split_config.language,
            chunk_size=split_config.chunk_size,
            chunk_overlap=split_config.chunk_overlap,
        )
        split_docs = splitter.split_text(
            content.content,
        )

        # Create Text objects with proper metadata inheritance
        results: List[Union[Text, Image]] = []
        current_position = 0

        for doc in split_docs:
            # Start with knowledge.metadata as base
            combined_metadata = {**knowledge.metadata}

            # Add Text.metadata from content (loader/previous parser stage)
            if content.metadata:
                combined_metadata.update(content.metadata)

            # Add processing information from current parser stage
            doc_start = content.content.find(doc, current_position)
            doc_end = (
                doc_start + len(doc) if doc_start != -1 else current_position + len(doc)
            )

            parser_metadata = {
                "chunk_start_position": (
                    doc_start if doc_start != -1 else current_position
                ),
                "chunk_end_position": doc_end,
                "chunk_length": len(doc),
                "parser_type": "base_code",
                "language": (
                    split_config.language.value
                    if hasattr(split_config.language, "value")
                    else str(split_config.language)
                ),
            }

            combined_metadata.update(parser_metadata)

            results.append(Text(content=doc, metadata=combined_metadata))
            current_position = doc_end

        return results

    async def batch_parse(
        self,
        knowledge: Knowledge,
        content_list: List[Text],
    ) -> List[ParseResult]:
        return [await self.parse(knowledge, content) for content in content_list]
