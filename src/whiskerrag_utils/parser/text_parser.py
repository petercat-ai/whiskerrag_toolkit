from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from whiskerrag_types.interface.splitter_interface import BaseSplitter
from whiskerrag_types.model.knowledge import TextSplitConfig
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.SPLITTER, "text")
class TextSplitter(BaseSplitter[TextSplitConfig, Text]):
    def split(self, content: Text, split_config: TextSplitConfig) -> List[Text]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=split_config.chunk_size,
            chunk_overlap=split_config.chunk_overlap,
            separators=split_config.separators,
            is_separator_regex=split_config.is_separator_regex,
            keep_separator=split_config.keep_separator or False,
        )
        split_texts = splitter.split_text(content.content)
        return [Text(content=text, metadata=content.metadata) for text in split_texts]

    def batch_split(
        self, content_list: List[Text], split_config: TextSplitConfig
    ) -> List[List[Text]]:
        return [self.split(content, split_config) for content in content_list]
