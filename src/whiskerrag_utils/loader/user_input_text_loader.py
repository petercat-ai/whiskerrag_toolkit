from whiskerrag_types.interface.loader_interface import BaseLoader
from whiskerrag_types.model.knowledge import KnowledgeSourceEnum, TextSourceConfig
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.KNOWLEDGE_LOADER, KnowledgeSourceEnum.USER_INPUT_TEXT)
class TextLoader(BaseLoader):

    async def load(self) -> str:
        if isinstance(self.knowledge.source_config, TextSourceConfig):
            return self.knowledge.source_config.text
        raise AttributeError(
            "source_config does not have a 'text' attribute for the current type."
        )
