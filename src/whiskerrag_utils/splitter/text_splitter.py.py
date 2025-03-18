from typing import List

from whiskerrag_types.interface.splitter_interface import BaseSplitter
from whiskerrag_types.model.knowledge import KnowledgeSplitConfig, KnowledgeTypeEnum
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.SPLITTER, KnowledgeTypeEnum.TEXT)
class TextSplitter(BaseSplitter):

    def split(self, content: str, config: KnowledgeSplitConfig) -> List[str]:
        return []
