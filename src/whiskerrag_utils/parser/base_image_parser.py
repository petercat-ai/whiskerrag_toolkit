from typing import List

from whiskerrag_types.interface.splitter_interface import BaseSplitter
from whiskerrag_types.model.multi_modal import Image
from whiskerrag_types.model.splitter import ImageSplitConfig
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.SPLITTER, "image")
class BaseTextParser(BaseSplitter[ImageSplitConfig, Image]):

    def split(self, content: Image, split_config: ImageSplitConfig) -> List[Image]:
        return [content]

    def batch_split(
        self, content_list: List[Image], split_config: ImageSplitConfig
    ) -> List[List[Image]]:
        return [self.split(content, split_config) for content in content_list]
