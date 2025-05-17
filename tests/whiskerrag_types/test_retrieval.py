import pytest
from pydantic import ValidationError

from whiskerrag_types.model.retrieval import RetrievalConfig, RetrievalRequest


class TestConfig(RetrievalConfig):
    additional_field: str = "test"


class AnotherConfig(RetrievalConfig):
    other_field: int = 42


def test_valid_request() -> None:
    """Test creating a valid request"""
    request = RetrievalRequest(
        content="test query",
        config={"type": "deep_retrieval", "embedding_model_name": "test-model"},
    )
    assert request.content == "test query"
    assert request.config.type == "deep_retrieval"
    assert request.config.model_dump()["embedding_model_name"] == "test-model"


def test_request_with_image() -> None:
    """Test request with image URL"""
    request = RetrievalRequest(
        content="test with image",
        image_url="https://example.com/image.jpg",
        config={"type": "deep_retrieval"},
    )
    assert request.image_url == "https://example.com/image.jpg"


def test_empty_content():
    """Test request with empty content should fail"""
    with pytest.raises(ValidationError) as exc_info:
        RetrievalRequest(content="", config={"type": "deep_retrieval"})
    assert "content must not be empty" in str(exc_info.value)


def test_retrieval_request_with_valid_subclass() -> None:
    """Test RetrievalRequest with valid config subclass."""
    # 测试有效的子类配置
    request = RetrievalRequest(
        content="test question", config=TestConfig(type="test_type")
    )
    assert isinstance(request.config, RetrievalConfig)
    assert isinstance(request.config, TestConfig)
    assert request.config.type == "test_type"
    assert request.config.additional_field == "test"


def test_retrieval_request_with_image() -> None:
    """Test RetrievalRequest with image URL."""
    # 测试包含图片URL的请求
    request = RetrievalRequest(
        content="describe this image",
        image_url="https://example.com/image.jpg",
        config=TestConfig(type="image_type"),
    )
    assert request.image_url == "https://example.com/image.jpg"


def test_retrieval_request_without_image() -> None:
    """Test RetrievalRequest without image URL."""
    # 测试不包含图片URL的请求
    request = RetrievalRequest(
        content="text only question", config=TestConfig(type="text_type")
    )
    assert request.image_url is None


def test_multiple_config_types() -> None:
    """Test RetrievalRequest with different config subclasses."""
    # 测试不同的配置子类
    request1 = RetrievalRequest(content="test1", config=TestConfig(type="type1"))
    request2 = RetrievalRequest(content="test2", config=AnotherConfig(type="type2"))

    assert isinstance(request1.config, TestConfig)
    assert isinstance(request2.config, AnotherConfig)
    assert request1.config.additional_field == "test"
    assert request2.config.other_field == 42


def test_invalid_content() -> None:
    """Test RetrievalRequest with empty content should fail."""
    # 测试空内容应该失败
    with pytest.raises(ValidationError):
        RetrievalRequest(content="", config=TestConfig(type="test_type"))


def test_invalid_config() -> None:
    """Test RetrievalRequest with invalid config should fail."""

    # 测试无效的配置应该失败
    class InvalidConfig:
        type = "invalid"

    with pytest.raises(ValidationError):
        RetrievalRequest(content="test", config=InvalidConfig())


@pytest.mark.parametrize(
    "image_url",
    ["https://example.com/image.jpg", "data:image/jpeg;base64,/9j/4AAQSkZJRg...", None],
)
def test_different_image_urls(image_url) -> None:
    """Test RetrievalRequest with different image URL formats."""
    # 测试不同格式的图片URL
    request = RetrievalRequest(
        content="test", image_url=image_url, config=TestConfig(type="test_type")
    )
    assert request.image_url == image_url
