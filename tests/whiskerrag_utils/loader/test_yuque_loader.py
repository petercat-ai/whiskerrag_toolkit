import asyncio
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from whiskerrag_types.model.knowledge import Knowledge, KnowledgeTypeEnum
from whiskerrag_types.model.knowledge_source import (
    OpenUrlSourceConfig,
    YuqueSourceConfig,
)
from whiskerrag_types.model.splitter import ImageSplitConfig, YuqueSplitConfig
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
                document_id="test_doc",
            ),
            split_config=YuqueSplitConfig(
                type="yuquedoc", separators=["\n\n"], is_separator_regex=False
            ),
            embedding_model_name="openai",
        )

    @patch("whiskerrag_utils.loader.yuque_loader.ExtendedYuqueLoader")
    def test_decompose_document(self, MockExtendedYuqueLoader):
        # Mock the loader and its methods
        mock_loader_instance = MockExtendedYuqueLoader.return_value
        mock_document = Document(
            page_content="This is a test document with an image. ![alt text](http://example.com/image.png)",
            metadata={"title": "Test Doc", "content_updated_at": "2023-01-01"},
        )
        mock_loader_instance.load_document_by_path.return_value = mock_document

        # Instantiate the loader
        loader = WhiskerYuqueLoader(self.knowledge)

        # Run decompose
        decomposed_knowledges = asyncio.run(loader.decompose())

        # Assertions - when decomposing a specific document, only images should be returned
        self.assertEqual(len(decomposed_knowledges), 1)

        # Check image knowledge (no document knowledge should be created)
        img_knowledge = decomposed_knowledges[0]
        self.assertEqual(img_knowledge.knowledge_type, KnowledgeTypeEnum.IMAGE)
        self.assertIn("image_0", img_knowledge.knowledge_name)
        self.assertIsInstance(img_knowledge.source_config, OpenUrlSourceConfig)
        self.assertEqual(
            img_knowledge.source_config.url, "http://example.com/image.png"
        )
        self.assertIsInstance(img_knowledge.split_config, ImageSplitConfig)

    @patch("whiskerrag_utils.loader.yuque_loader.ExtendedYuqueLoader")
    def test_decompose_book(self, MockExtendedYuqueLoader):
        # Mock the loader and its methods
        mock_loader_instance = MockExtendedYuqueLoader.return_value
        mock_loader_instance.get_book_documents_by_path.return_value = [
            {"slug": "doc1"},
            {"slug": "doc2"},
        ]
        mock_document1 = Document(
            page_content="Doc 1 content",
            metadata={"title": "Doc 1", "content_updated_at": "2023-01-01"},
        )
        mock_document2 = Document(
            page_content="Doc 2 content with image ![img](http://a.com/b.png)",
            metadata={"title": "Doc 2", "content_updated_at": "2023-01-02"},
        )
        mock_loader_instance.load_document_by_path.side_effect = [
            mock_document1,
            mock_document2,
        ]

        # Adjust knowledge to be a book
        self.knowledge.source_config.document_id = None
        loader = WhiskerYuqueLoader(self.knowledge)

        # Run decompose
        decomposed_knowledges = asyncio.run(loader.decompose())

        # Assertions - when decomposing a book, only documents should be returned (no images)
        self.assertEqual(len(decomposed_knowledges), 2)  # only doc1 and doc2

        # Check first document knowledge
        self.assertEqual(
            decomposed_knowledges[0].knowledge_name, "Test Yuque Doc/Doc 1"
        )
        self.assertEqual(
            decomposed_knowledges[0].knowledge_type, KnowledgeTypeEnum.YUQUEDOC
        )

        # Check second document knowledge
        self.assertEqual(
            decomposed_knowledges[1].knowledge_name, "Test Yuque Doc/Doc 2"
        )
        self.assertEqual(
            decomposed_knowledges[1].knowledge_type, KnowledgeTypeEnum.YUQUEDOC
        )

    @patch("whiskerrag_utils.loader.yuque_loader.ExtendedYuqueLoader")
    def test_decompose_book_then_document_images(self, MockExtendedYuqueLoader):
        """测试先从 book 获取文档知识点，然后对文档进行 decompose 获取图片"""
        # Mock the loader and its methods
        mock_loader_instance = MockExtendedYuqueLoader.return_value
        mock_loader_instance.get_book_documents_by_path.return_value = [
            {"slug": "doc1"},
            {"slug": "doc2"},
        ]
        mock_document1 = Document(
            page_content="Doc 1 content",
            metadata={
                "title": "Doc 1",
                "content_updated_at": "2023-01-01",
                "slug": "doc1",
            },
        )
        mock_document2 = Document(
            page_content="Doc 2 content with image ![img](http://example.com/image2.png)",
            metadata={
                "title": "Doc 2",
                "content_updated_at": "2023-01-02",
                "slug": "doc2",
            },
        )
        mock_loader_instance.load_document_by_path.side_effect = [
            mock_document1,
            mock_document2,
            mock_document2,  # 第二次调用时返回 doc2（用于获取图片）
        ]

        # Step 1: 先从 book 获取文档知识点
        self.knowledge.source_config.document_id = None
        book_loader = WhiskerYuqueLoader(self.knowledge)
        doc_knowledges = asyncio.run(book_loader.decompose())

        # 验证获取到了文档知识点
        self.assertEqual(len(doc_knowledges), 2)
        self.assertEqual(doc_knowledges[0].knowledge_type, KnowledgeTypeEnum.YUQUEDOC)
        self.assertEqual(doc_knowledges[1].knowledge_type, KnowledgeTypeEnum.YUQUEDOC)

        # Step 2: 对其中一个包含图片的文档进行 decompose 获取图片
        doc2_knowledge = doc_knowledges[1]  # Doc 2 包含图片
        doc_loader = WhiskerYuqueLoader(doc2_knowledge)
        image_knowledges = asyncio.run(doc_loader.decompose())

        # 验证获取到了图片知识点
        self.assertEqual(len(image_knowledges), 1)
        img_knowledge = image_knowledges[0]
        self.assertEqual(img_knowledge.knowledge_type, KnowledgeTypeEnum.IMAGE)
        self.assertIn("image_0", img_knowledge.knowledge_name)
        self.assertIsInstance(img_knowledge.source_config, OpenUrlSourceConfig)
        self.assertEqual(
            img_knowledge.source_config.url, "http://example.com/image2.png"
        )
        self.assertIsInstance(img_knowledge.split_config, ImageSplitConfig)


if __name__ == "__main__":
    unittest.main()
