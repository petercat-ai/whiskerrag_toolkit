import re
from typing import List

from langchain_text_splitters import CharacterTextSplitter

from whiskerrag_types.interface.splitter_interface import BaseSplitter
from whiskerrag_types.model.knowledge import KnowledgeTypeEnum, TextSplitConfig
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.SPLITTER, KnowledgeTypeEnum.TEXT)
class TextSplitter(BaseSplitter[TextSplitConfig, Text]):
    def split(self, content: str, split_config: TextSplitConfig) -> List[Text]:
        config_dict = split_config.model_dump(mode="json")
        separators = config_dict["separators"] or ["\n\n"]
        separator_pattern = "|".join(map(re.escape, separators))
        splitter = CharacterTextSplitter(
            chunk_size=config_dict["chunk_size"],
            chunk_overlap=config_dict["chunk_overlap"],
            separator=separator_pattern,
            is_separator_regex=True,
            keep_separator=config_dict["keep_separator"],
            strip_whitespace=config_dict["strip_whitespace"],
        )
        return splitter.split_text(content)

    def batch_split(
        self, content: List[str], split_config: TextSplitConfig
    ) -> List[List[str]]:
        return [self.split(text, split_config) for text in content]
