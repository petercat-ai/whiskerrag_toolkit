from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from whiskerrag_types.interface.parser_interface import BaseParser, ParseResult
from whiskerrag_types.model.knowledge import Knowledge
from whiskerrag_types.model.multi_modal import Text
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
        # TODO:metadata must have position in the content
        return [Text(content=doc, metadata=content.metadata) for doc in split_docs]

    async def batch_parse(
        self,
        knowledge: Knowledge,
        content_list: List[Text],
    ) -> List[ParseResult]:
        return [await self.parse(knowledge, content) for content in content_list]
