import os
import tempfile

import pytest

from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
    Knowledge,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
)
from whiskerrag_types.model.knowledge_source import OpenUrlSourceConfig
from whiskerrag_utils.loader.cloud_storage_text_loader import CloudStorageTextLoader


class TestReadFileContent:
    """测试 read_file_content 方法的编码兼容性"""

    def setup_method(self):
        """测试前的准备工作"""
        self.knowledge_data = {
            "space_id": "test_space",
            "knowledge_type": KnowledgeTypeEnum.TEXT,
            "knowledge_name": "Test Knowledge",
            "source_type": KnowledgeSourceEnum.CLOUD_STORAGE_TEXT,
            "source_config": OpenUrlSourceConfig(url="https://example.com/test.txt"),
            "embedding_model_name": EmbeddingModelEnum.OPENAI,
            "split_config": {
                "type": "text",
                "chunk_size": 500,
                "chunk_overlap": 100,
            },
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
        }

    def test_read_utf8_file(self):
        """测试读取UTF-8编码文件"""
        knowledge = Knowledge(**self.knowledge_data)
        loader = CloudStorageTextLoader(knowledge)

        # 创建UTF-8编码的测试文件
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
            test_content = "Hello, 世界! This is UTF-8 content."
            f.write(test_content)
            temp_path = f.name

        try:
            content = loader.read_file_content(temp_path)
            assert content == test_content
        finally:
            os.unlink(temp_path)

    def test_read_gbk_file(self):
        """测试读取GBK编码文件"""
        knowledge = Knowledge(**self.knowledge_data)
        loader = CloudStorageTextLoader(knowledge)

        # 创建GBK编码的测试文件
        with tempfile.NamedTemporaryFile(mode="w", encoding="gbk", delete=False) as f:
            test_content = "你好，世界！这是GBK编码的内容。"
            f.write(test_content)
            temp_path = f.name

        try:
            content = loader.read_file_content(temp_path)
            assert "你好" in content
            assert "GBK编码" in content
        finally:
            os.unlink(temp_path)

    def test_read_gb2312_file(self):
        """测试读取GB2312编码文件"""
        knowledge = Knowledge(**self.knowledge_data)
        loader = CloudStorageTextLoader(knowledge)

        # 创建GB2312编码的测试文件
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="gb2312", delete=False
        ) as f:
            test_content = "简体中文测试内容"
            f.write(test_content)
            temp_path = f.name

        try:
            content = loader.read_file_content(temp_path)
            assert "简体中文" in content
        finally:
            os.unlink(temp_path)

    def test_read_binary_file_with_ignore_errors(self):
        """测试读取包含无法解码字符的文件"""
        knowledge = Knowledge(**self.knowledge_data)
        loader = CloudStorageTextLoader(knowledge)

        # 创建包含二进制数据的文件
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            # 写入一些文本和无效的字节序列
            f.write(b"Valid text content\x80\x81\x82")
            temp_path = f.name

        try:
            content = loader.read_file_content(temp_path)
            # 应该能够读取到有效文本部分
            assert "Valid text content" in content
        finally:
            os.unlink(temp_path)

    def test_read_empty_file(self):
        """测试读取空文件"""
        knowledge = Knowledge(**self.knowledge_data)
        loader = CloudStorageTextLoader(knowledge)

        # 创建空文件
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
            temp_path = f.name

        try:
            content = loader.read_file_content(temp_path)
            assert content == ""
        finally:
            os.unlink(temp_path)

    def test_read_large_file(self):
        """测试读取大文件"""
        knowledge = Knowledge(**self.knowledge_data)
        loader = CloudStorageTextLoader(knowledge)

        # 创建大文件
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
            large_content = "Large file content. " * 10000
            f.write(large_content)
            temp_path = f.name

        try:
            content = loader.read_file_content(temp_path)
            assert len(content) == len(large_content)
            assert "Large file content" in content
        finally:
            os.unlink(temp_path)

    def test_read_nonexistent_file(self):
        """测试读取不存在的文件"""
        knowledge = Knowledge(**self.knowledge_data)
        loader = CloudStorageTextLoader(knowledge)

        nonexistent_path = "/tmp/nonexistent_file_12345.txt"

        # 应该抛出异常
        with pytest.raises(Exception) as exc_info:
            loader.read_file_content(nonexistent_path)

        # 检查异常类型和消息
        assert exc_info.type in [FileNotFoundError, OSError, Exception]
        assert any(
            msg in str(exc_info.value).lower()
            for msg in [
                "no such file",
                "failed to read file",
                "cannot find",
                "does not exist",
            ]
        )


if __name__ == "__main__":
    pytest.main([__file__])
