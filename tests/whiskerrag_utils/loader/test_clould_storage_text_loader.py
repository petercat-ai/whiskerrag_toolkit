import os
import tempfile
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
    Knowledge,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
)
from whiskerrag_types.model.knowledge_source import OpenUrlSourceConfig
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils.loader.cloud_storage_text_loader import CloudStorageTextLoader


@pytest.mark.asyncio
async def test_cloud_storage_text_loader() -> None:
    """测试云存储文本加载器"""
    # Arrange
    text_url = "https://xxx"
    url = text_url
    config = OpenUrlSourceConfig(url=url)
    knowledge_data = {
        "space_id": "test_space",
        "knowledge_type": KnowledgeTypeEnum.JSON,
        "knowledge_name": "Test Knowledge",
        "source_type": KnowledgeSourceEnum.CLOUD_STORAGE_TEXT,
        "source_config": config,
        "embedding_model_name": EmbeddingModelEnum.OPENAI,
        "split_config": {
            "type": "json",
            "chunk_size": 500,
            "chunk_overlap": 100,
        },
        "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
    }

    knowledge = Knowledge(**knowledge_data)
    loader = CloudStorageTextLoader(knowledge)

    # 创建测试内容
    temp_content = b"This is test content"

    # 创建模拟响应对象
    mock_response = MagicMock()
    mock_response.headers = {
        "content-type": "text/plain",
        "content-length": str(len(temp_content)),
        "last-modified": "Wed, 21 Oct 2023 07:28:00 GMT",
        "etag": '"123456"',
        "content-disposition": 'attachment; filename="test.txt"',
    }
    mock_response.raise_for_status.return_value = None

    # 模拟 iter_content 方法
    mock_response.iter_content.return_value = [temp_content]

    # 实现上下文管理器协议
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None

    # 使用 patch 替换 requests.get
    with patch("requests.get", return_value=mock_response) as mock_get:
        # Act
        results: List[Text] = await loader.load()

        # Assert
        assert results is not None
        assert len(results) > 0

        for text in results:
            # 验证内容
            assert isinstance(text, Text)
            assert text.content is not None
            assert len(text.content) > 0
            # 验证元数据
            assert text.metadata is not None
            assert isinstance(text.metadata, dict)
            # 验证元数据字段
            assert text.metadata.get("content_type") == "text/plain"
            assert text.metadata.get("original_filename") == "test.txt"

        # 验证 mock 是否被正确调用
        mock_get.assert_called_once_with(url, stream=True)
        mock_response.iter_content.assert_called_with(chunk_size=8192)

    # 确保任何临时文件都被清理
    temp_dir = tempfile.gettempdir()
    for f in os.listdir(temp_dir):
        if f.startswith("download_"):
            try:
                os.unlink(os.path.join(temp_dir, f))
            except OSError:
                pass


@pytest.mark.asyncio
async def test_cloud_storage_text_loader_invalid_url() -> None:
    """测试无效URL的情况"""
    # Arrange
    invalid_url = (
        f"https://a-real-invalid-url-that-does-not-exist-{time.time()}.com/file.txt"
    )
    knowledge_data = {
        "space_id": "test_space",
        "knowledge_type": KnowledgeTypeEnum.JSON,
        "knowledge_name": "Test Knowledge",
        "source_type": KnowledgeSourceEnum.CLOUD_STORAGE_TEXT,
        "source_config": OpenUrlSourceConfig(url=invalid_url),
        "embedding_model_name": EmbeddingModelEnum.OPENAI,
        "split_config": {
            "type": "json",
            "chunk_size": 500,
            "chunk_overlap": 100,
        },
        "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
    }

    with pytest.raises(Exception) as exc_info:
        knowledge = Knowledge(**knowledge_data)
        loader = CloudStorageTextLoader(knowledge)
        await loader.load()
    assert "Failed to load content from cloud storage" in str(exc_info.value)
