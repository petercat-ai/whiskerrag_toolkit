from whiskerrag_types.interface.embed_interface import BaseEmbedding
from whiskerrag_types.interface.loader_interface import BaseLoader
from whiskerrag_types.model.knowledge import EmbeddingModelEnum, KnowledgeSourceEnum
from whiskerrag_utils import RegisterTypeEnum, get_register


# poetry run pytest tests/whiskerrag_utils/test_registry.py
class TestRegister:
    def test_get_github_loader(self):
        github_loader = get_register(
            RegisterTypeEnum.KNOWLEDGE_LOADER, KnowledgeSourceEnum.GITHUB_REPO
        )
        assert issubclass(github_loader, BaseLoader)

    def test_get_openai_embedding(self):
        embedding = get_register(RegisterTypeEnum.EMBEDDING, EmbeddingModelEnum.OPENAI)
        assert issubclass(embedding, BaseEmbedding)
