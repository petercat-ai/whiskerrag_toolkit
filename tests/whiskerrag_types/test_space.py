from whiskerrag_types.model.space import Space, SpaceCreate, SpaceResponse


class TestSpace:
    def test_space_create_initialization(self) -> None:
        data = {
            "space_name": "test_space",
            "description": "this is a test",
        }
        space_create = SpaceCreate(**data)
        assert space_create.space_name == "test_space"
        assert space_create.description == "this is a test"
        assert space_create.metadata == {}
        assert space_create.model_dump() == {
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
