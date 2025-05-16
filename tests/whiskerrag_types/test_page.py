from typing import Optional

import pytest
from pydantic import BaseModel, ValidationError

from whiskerrag_types.model.page import BasePageParams, PageParams, QueryParams


class DummyModel(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


class TestPageParams:
    def test_default_values(self):
        params = PageParams[DummyModel]()
        assert params.page == 1
        assert params.page_size == 10
        assert params.order_by is None
        assert params.order_direction == "asc"
        assert params.eq_conditions is None

    def test_offset_and_limit(self):
        params = PageParams[DummyModel](page=2, page_size=20)
        assert params.offset == 20  # (2-1) * 20
        assert params.limit == 20

    def test_valid_eq_conditions(self):
        params = PageParams[DummyModel](
            eq_conditions={"name": "John", "age": 25, "email": "john@example.com"}
        )
        assert params.eq_conditions == {
            "name": "John",
            "age": 25,
            "email": "john@example.com",
        }

    def test_invalid_eq_conditions(self):
        with pytest.raises(ValueError) as exc_info:
            PageParams[DummyModel](
                eq_conditions={"invalid_field": "value", "name": "John"}
            )
        assert "Invalid keys in eq_conditions" in str(exc_info.value)

    def test_eq_conditions_with_none(self):
        params = PageParams[DummyModel](eq_conditions=None)
        assert params.eq_conditions is None

    def test_page_validation(self):
        with pytest.raises(ValidationError):
            PageParams[DummyModel](page=0)  # page 必须 >= 1

    def test_page_size_validation(self):
        with pytest.raises(ValidationError):
            PageParams[DummyModel](page_size=0)  # page_size 必须 >= 1
        with pytest.raises(ValidationError):
            PageParams[DummyModel](page_size=1001)  # page_size 必须 <= 1000

    def test_order_direction_validation(self):
        params = PageParams[DummyModel](order_direction="desc")
        assert params.order_direction == "desc"
        params = PageParams[DummyModel](order_direction="asc")
        assert params.order_direction == "asc"

    def test_partial_eq_conditions(self):
        params = PageParams[DummyModel](eq_conditions={"name": "John"})
        assert params.eq_conditions == {"name": "John"}

    def test_empty_eq_conditions(self):
        params = PageParams[DummyModel](eq_conditions={})
        assert params.eq_conditions == {}

    def test_generic_type_validation(self):
        with pytest.raises(ValueError):
            PageParams[DummyModel](eq_conditions={"invalid_field": "test"})

    def test_generic_type_with_typever(self):
        from typing import TypeVar

        T = TypeVar("T")
        params = PageParams[T](eq_conditions={"field": "value"})
        assert params.eq_conditions == {"field": "value"}

    def test_base_page_params(self):
        params = BasePageParams()
        assert params.page == 1
        assert params.page_size == 10

    def test_query_params(self):
        params = QueryParams[DummyModel]()
        assert params.order_by is None
        assert params.order_direction == "asc"
        assert params.eq_conditions is None
