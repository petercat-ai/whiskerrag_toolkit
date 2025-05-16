from typing import List

from whiskerrag_types.interface.parser_interface import BaseParser, SplitResult
from whiskerrag_types.model.knowledge import Knowledge
from whiskerrag_types.model.multi_modal import Image
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.SPLITTER, "image")
class BaseTextParser(BaseParser[Image]):

    def parse(
        self,
        knowledge: Knowledge,
        content: Image,
    ) -> SplitResult:
        return [content]

    def batch_parse(
        self,
        knowledge: Knowledge,
        content_list: List[Image],
    ) -> List[SplitResult]:
        return [self.parse(knowledge, content) for content in content_list]
