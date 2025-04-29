from typing import List

from langchain_text_splitters import MarkdownTextSplitter

from whiskerrag_types.interface.parser_interface import BaseParser
from whiskerrag_types.model.knowledge import MarkdownParseConfig
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.PARSER, "markdown")
class MarkdownSplitter(BaseParser[MarkdownParseConfig, Text]):

    def split(self, content: str, split_config: MarkdownParseConfig) -> List[str]:
        splitter = MarkdownTextSplitter(
            chunk_size=split_config.chunk_size,
            chunk_overlap=split_config.chunk_overlap,
        )
        return splitter.split_text(content)

    def batch_split(
        self, content: List[str], split_config: MarkdownParseConfig
    ) -> List[List[str]]:
        return [self.split(text, split_config) for text in content]
