import re
from typing import List

from langchain_text_splitters import CharacterTextSplitter

from whiskerrag_types.interface.splitter_interface import BaseSplitter
from whiskerrag_types.model.knowledge import KnowledgeTypeEnum, TextSplitConfig
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils.registry import RegisterTypeEnum, register


def separators_to_regex(separators: list[str]) -> str:
    escaped_separators = [re.escape(sep) for sep in separators]
    pattern = "|".join(escaped_separators)
    return pattern


@register(RegisterTypeEnum.SPLITTER, KnowledgeTypeEnum.TEXT)
class TextSplitter(BaseSplitter[TextSplitConfig, Text]):
    def split(self, content: str, split_config: TextSplitConfig) -> List[Text]:
        splitter = CharacterTextSplitter(
            chunk_size=split_config.chunk_size,
            chunk_overlap=split_config.chunk_overlap,
            separator=separators_to_regex(split_config.separators or ["\n\n"]),
            is_regex=True,
            keep_separator=split_config.keep_separator,
            strip_whitespace=split_config.strip_whitespace,
        )
        return splitter.split_text(content)

    def batch_split(
        self, content: List[str], split_config: TextSplitConfig
    ) -> List[List[str]]:
        return [self.split(text, split_config) for text in content]
