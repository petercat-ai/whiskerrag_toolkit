import json
from typing import List

from langchain_text_splitters import RecursiveJsonSplitter

from whiskerrag_types.interface.parser_interface import BaseParser
from whiskerrag_types.model.knowledge import JSONParseConfig
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils.registry import RegisterTypeEnum, register


@register(RegisterTypeEnum.PARSER, "json")
class JSONParser(BaseParser[JSONParseConfig, Text]):

    def split(self, content: str, split_config: JSONParseConfig) -> List[Text]:
        """Splits JSON content into smaller chunks based on the provided configuration."""
        json_content = {}
        try:
            json_content = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON content provided for splitting.")
        parser = RecursiveJsonSplitter(
            max_chunk_size=split_config.max_chunk_size,
            min_chunk_size=split_config.min_chunk_size,
        )
        return parser.split_text(json_content)

    def batch_split(
        self, content: List[str], split_config: JSONParseConfig
    ) -> List[List[str]]:
        return [self.split(text, split_config) for text in content]
