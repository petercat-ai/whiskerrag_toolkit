import pytest
from pydantic import ValidationError

from whiskerrag_types.model.space import Space, SpaceCreate, SpaceResponse


class TestSpace:

    def test_space_create_valid(self) -> None:
        # 测试有效的数据
        valid_data = {
            "space_name": "test space",
            "space_id": "test-space-123",
            "description": "This is a test space",
            "metadata": {"model": "gpt-3.5"},
        }
        space_create = SpaceCreate(**valid_data)

        space = Space(
            **space_create.model_dump(),
            tenant_id="38fbd88b-e869-489c-9142-e4ea2c2261db"
        )
        assert space_create.space_name == "test space"
        assert space_create.space_id == "test-space-123"
        assert space_create.description == "This is a test space"
        assert space_create.metadata == {"model": "gpt-3.5"}
        assert space.space_id == "test-space-123"

    def test_space_create_without_space_id(self) -> None:
        # 测试可选的 space_id
        data = {
            "space_name": "test space",
            "description": "This is a test space",
            "metadata": {},
        }
        space_create = SpaceCreate(**data)
        space = Space(
            **space_create.model_dump(),
            tenant_id="38fbd88b-e869-489c-9142-e4ea2c2261db"
        )
        assert space.space_id is not None

    def test_space_create_invalid_space_id(self) -> None:
        # 测试无效的 space_id
        invalid_cases = [
            # 特殊字符
            {"space_name": "test", "space_id": "test@space", "description": "test"},
            # 超长
            {"space_name": "test", "space_id": "a" * 65, "description": "test"},
            # 空字符串
            {"space_name": "test", "space_id": "", "description": "test"},
        ]

        for invalid_data in invalid_cases:
            with pytest.raises(ValidationError):
                SpaceCreate(**invalid_data)

    def test_space_create_invalid_space_name(self) -> None:
        # 测试超长的 space_name
        with pytest.raises(ValidationError):
            SpaceCreate(space_name="a" * 65, description="test")

    def test_space_create_invalid_description(self) -> None:
        # 测试超长的 description
        with pytest.raises(ValidationError):
            SpaceCreate(space_name="test", description="a" * 256)

    def test_space_create_metadata(self) -> None:
        # 测试各种类型的 metadata
        test_cases = [
            {},  # 空字典
            {"key": "value"},  # 简单键值对
            {"nested": {"key": "value"}},  # 嵌套字典
            {"list": [1, 2, 3]},  # 包含列表
            {"number": 42, "boolean": True},  # 混合类型
        ]

        for metadata in test_cases:
            space = SpaceCreate(
                space_name="test", description="test", metadata=metadata
            )
            assert space.metadata == metadata

    def test_space_create_initialization(self) -> None:
        data = {
            "space_name": "test_space",
            "description": "this is a test",
        }
        space_create = SpaceCreate(**data)
        print("[]", space_create.model_dump())
        assert space_create.space_name == "test_space"
        assert space_create.description == "this is a test"
        assert space_create.metadata == {}
        assert space_create.model_dump() == {
            "space_name": "test_space",
            "space_id": None,
            "description": "this is a test",
            "metadata": {},
        }
        assert space_create.model_dump(exclude_none=True) == {
            "space_name": "test_space",
            "description": "this is a test",
            "metadata": {},
        }

    def test_space_initialization(self) -> None:
        data = {
            "space_name": "test_space",
            "description": "this is a test",
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
        }
        space = Space(**data)
        assert space.space_name == "test_space"
        assert space.description == "this is a test"
        assert space.tenant_id == "38fbd88b-e869-489c-9142-e4ea2c2261db"
        assert space.space_id is not None
        assert space.created_at is not None
        assert space.updated_at is not None

    def test_space_dump(self) -> None:
        data = {
            "space_name": "test_space",
            "description": "this is a test",
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
            "gmt_create": "2023-01-01T00:00:00Z",
            "gmt_modified": "2023-01-01T00:00:00Z",
        }
        space = Space(**data).model_dump()
        assert space["created_at"] == "2023-01-01T00:00:00.000000Z"
        assert "gmt_create" not in space

    def test_space_update(self) -> None:
        data = {
            "space_name": "test_space",
            "description": "this is a test",
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
        }
        space = Space(**data)
        updated_data = {"description": "Updated desc"}
        space.update(**updated_data)
        assert space.description == "Updated desc"

    def test_space_response(self) -> None:
        data = {
            "space_name": "test_space",
            "description": "this is a test",
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c2261db",
        }
        space = Space(**data)
        response = SpaceResponse(**space.model_dump())
        updated_data = {"storage_size": 1231313, "knowledge_count": 20}
        response.update(**updated_data)
        assert response.storage_size == 1231313
        assert response.knowledge_count == 20
