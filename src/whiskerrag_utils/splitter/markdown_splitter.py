from typing import List

from whiskerrag_types.interface.splitter_interface import BaseSplitter
from whiskerrag_types.model.knowledge import KnowledgeSplitConfig, KnowledgeTypeEnum
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.SPLITTER, KnowledgeTypeEnum.MARKDOWN)
class MarkdownSplitter(BaseSplitter):

    def split(self, content: str, config: KnowledgeSplitConfig) -> List[str]:
        return []
