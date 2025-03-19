from typing import List

from langchain_text_splitters import CharacterTextSplitter

from whiskerrag_types.interface.splitter_interface import BaseSplitter
from whiskerrag_types.model.knowledge import KnowledgeTypeEnum, TextSplitConfig
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.SPLITTER, KnowledgeTypeEnum.TEXT)
class TextSplitter(BaseSplitter[TextSplitConfig, Text]):
    def split(self, content: str, split_config: TextSplitConfig) -> List[Text]:
        splitter = CharacterTextSplitter(
            chunk_size=split_config.chunk_size,
            chunk_overlap=split_config.chunk_overlap,
        )
        return splitter.split_text(content)

    def batch_split(
        self, content: List[str], split_config: TextSplitConfig
    ) -> List[List[str]]:
        return [self.split(text, split_config) for text in content]
