import importlib
import os
import sys

from whiskerrag_types.interface.embed_interface import BaseEmbedding
from whiskerrag_types.interface.loader_interface import BaseLoader
from whiskerrag_types.model.knowledge import EmbeddingModelEnum, KnowledgeSourceEnum
from whiskerrag_utils import RegisterTypeEnum, get_register
from whiskerrag_utils.registry import init_register


# poetry run pytest tests/whiskerrag_utils/test_registry.py
class TestRegister:
    def test_get_github_loader(self) -> None:
        init_register("whiskerrag_utils.loader")
        github_loader = get_register(
            RegisterTypeEnum.KNOWLEDGE_LOADER, KnowledgeSourceEnum.GITHUB_FILE
        )
        assert issubclass(github_loader, BaseLoader)

    def test_get_user_input_loader(self) -> None:
        init_register("whiskerrag_utils.loader")
        text_loader = get_register(
            RegisterTypeEnum.KNOWLEDGE_LOADER, KnowledgeSourceEnum.USER_INPUT_TEXT
        )
        assert issubclass(text_loader, BaseLoader)

    def test_get_openai_embedding(self) -> None:
        # Set environment variable before any registration
        os.environ["OPENAI_API_KEY"] = "test_openai_api_key"

        # Force re-registration by clearing the loaded packages and reimporting
        from whiskerrag_utils.registry import _loaded_packages

        _loaded_packages.discard("whiskerrag_utils.embedding")

        # Force reimport of the embedding module
        module_name = "whiskerrag_utils.embedding.openai"
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])

        init_register("whiskerrag_utils.embedding")
        embedding = get_register(RegisterTypeEnum.EMBEDDING, EmbeddingModelEnum.OPENAI)
        assert issubclass(embedding, BaseEmbedding)
