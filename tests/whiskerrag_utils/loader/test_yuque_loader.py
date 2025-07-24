import asyncio
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from whiskerrag_types.model.knowledge import Knowledge, KnowledgeTypeEnum
from whiskerrag_types.model.knowledge_source import YuqueSourceConfig, OpenUrlSourceConfig
from whiskerrag_types.model.splitter import YuqueSplitConfig, ImageSplitConfig
from whiskerrag_utils.loader.yuque_loader import WhiskerYuqueLoader


class TestWhiskerYuqueLoader(unittest.TestCase):

    def setUp(self):
        self.knowledge = Knowledge(
            knowledge_id="test_knowledge",
            space_id="test_space",
            tenant_id="test_tenant",
            knowledge_name="Test Yuque Doc",
            knowledge_type=KnowledgeTypeEnum.YUQUEDOC,
            source_type="yuque",
            source_config=YuqueSourceConfig(
                auth_info="test_token",
                group_login="test_group",
                book_slug="test_book",
                document_id="test_doc"
            ),
            split_config=YuqueSplitConfig(type="yuquedoc", separators=['\n\n'], is_separator_regex=False),
            embedding_model_name="openai"
        )

    @patch('whiskerrag_utils.loader.yuque_loader.ExtendedYuqueLoader')
    def test_decompose_document(self, MockExtendedYuqueLoader):
        # Mock the loader and its methods
        mock_loader_instance = MockExtendedYuqueLoader.return_value
        mock_document = Document(
            page_content="This is a test document with an image. ![alt text](http://example.com/image.png)",
            metadata={'title': 'Test Doc', 'content_updated_at': '2023-01-01'}
        )
        mock_loader_instance.load_document_by_path.return_value = mock_document

        # Instantiate the loader
        loader = WhiskerYuqueLoader(self.knowledge)

        # Run decompose
        decomposed_knowledges = asyncio.run(loader.decompose())

        # Assertions
        self.assertEqual(len(decomposed_knowledges), 2)

        # Check document knowledge
        doc_knowledge = decomposed_knowledges[0]
        self.assertEqual(doc_knowledge.knowledge_type, KnowledgeTypeEnum.YUQUEDOC)
        self.assertIn('Test Doc', doc_knowledge.knowledge_name)

        # Check image knowledge
        img_knowledge = decomposed_knowledges[1]
        self.assertEqual(img_knowledge.knowledge_type, KnowledgeTypeEnum.IMAGE)
        self.assertIn('image_0', img_knowledge.knowledge_name)
        self.assertIsInstance(img_knowledge.source_config, OpenUrlSourceConfig)
        self.assertEqual(img_knowledge.source_config.url, 'http://example.com/image.png')
        self.assertIsInstance(img_knowledge.split_config, ImageSplitConfig)

    @patch('whiskerrag_utils.loader.yuque_loader.ExtendedYuqueLoader')
    def test_decompose_book(self, MockExtendedYuqueLoader):
        # Mock the loader and its methods
        mock_loader_instance = MockExtendedYuqueLoader.return_value
        mock_loader_instance.get_book_documents_by_path.return_value = [
            {'slug': 'doc1'},
            {'slug': 'doc2'}
        ]
        mock_document1 = Document(
            page_content="Doc 1 content",
            metadata={'title': 'Doc 1', 'content_updated_at': '2023-01-01'}
        )
        mock_document2 = Document(
            page_content="Doc 2 content with image ![img](http://a.com/b.png)",
            metadata={'title': 'Doc 2', 'content_updated_at': '2023-01-02'}
        )
        mock_loader_instance.load_document_by_path.side_effect = [mock_document1, mock_document2]

        # Adjust knowledge to be a book
        self.knowledge.source_config.document_id = None
        loader = WhiskerYuqueLoader(self.knowledge)

        # Run decompose
        decomposed_knowledges = asyncio.run(loader.decompose())

        # Assertions
        self.assertEqual(len(decomposed_knowledges), 3) # doc1, doc2, image_in_doc2
        self.assertEqual(decomposed_knowledges[0].knowledge_name, "Test Yuque Doc/Doc 1")
        self.assertEqual(decomposed_knowledges[1].knowledge_name, "Test Yuque Doc/Doc 2")
        self.assertEqual(decomposed_knowledges[2].knowledge_name, "Test Yuque Doc/Doc 2/image_0")
        self.assertEqual(decomposed_knowledges[2].knowledge_type, KnowledgeTypeEnum.IMAGE)


if __name__ == '__main__':
    unittest.main()
