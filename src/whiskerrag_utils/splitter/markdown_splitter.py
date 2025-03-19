from typing import List

from langchain_text_splitters import MarkdownTextSplitter

from whiskerrag_types.interface.splitter_interface import BaseSplitter
from whiskerrag_types.model.knowledge import KnowledgeTypeEnum, MarkdownSplitConfig
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.SPLITTER, KnowledgeTypeEnum.MARKDOWN)
class MarkdownSplitter(BaseSplitter[MarkdownSplitConfig, Text]):

    def split(self, content: str, split_config: MarkdownSplitConfig) -> List[str]:
        splitter = MarkdownTextSplitter(
            chunk_size=split_config.chunk_size,
            chunk_overlap=split_config.chunk_overlap,
        )
        return splitter.split_text(content)

    def batch_split(
        self, content: List[str], split_config: MarkdownSplitConfig
    ) -> List[List[str]]:
        return [self.split(text, split_config) for text in content]
