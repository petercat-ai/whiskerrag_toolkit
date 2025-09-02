from typing import Optional

import pytest
from pydantic import BaseModel, Field, ValidationError

from whiskerrag_types.model.page import PageParams, PageQueryParams, QueryParams


class DummyModel(BaseModel):
    name: str
    age: int
    email: Optional[str] = None
    created_at: Optional[str] = Field(default=None, alias="gmt_create")
    updated_at: Optional[str] = Field(default=None, alias="gmt_modified")


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
        assert "Invalid fields found: {'invalid_field'}" in str(exc_info.value)

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
        params = PageQueryParams()
        assert params.page == 1
        assert params.page_size == 10

    def test_query_params(self):
        params = QueryParams[DummyModel]()
        assert params.order_by is None
        assert params.order_direction == "asc"
        assert params.eq_conditions is None

    def test_alias_field_validation_issue(self):
        """Test that alias fields are now properly validated after the fix"""
        # This test demonstrates that the issue has been fixed: gmt_create is now valid as it's an alias for created_at
        # The fix ensures that _validate_fields_against_model considers field aliases
        params = PageParams[DummyModel](
            eq_conditions={"gmt_create": "2023-01-01T00:00:00Z"}
        )
        assert params.eq_conditions == {"gmt_create": "2023-01-01T00:00:00Z"}

        # Test that the fix works for both alias and regular field names
        params = PageParams[DummyModel](
            eq_conditions={
                "gmt_create": "2023-01-01T00:00:00Z",
                "created_at": "2023-01-02T00:00:00Z",  # Both should work
            }
        )
        assert params.eq_conditions == {
            "gmt_create": "2023-01-01T00:00:00Z",
            "created_at": "2023-01-02T00:00:00Z",
        }

    def test_alias_field_validation_fixed(self):
        """Test that alias fields are now properly validated after the fix"""
        # This should now work with the fix
        params = PageQueryParams[DummyModel](
            eq_conditions={"gmt_create": "2023-01-01T00:00:00Z"}
        )
        assert params.eq_conditions == {"gmt_create": "2023-01-01T00:00:00Z"}

        # Test with both alias and regular field names
        params = PageQueryParams[DummyModel](
            eq_conditions={
                "gmt_create": "2023-01-01T00:00:00Z",
                "gmt_modified": "2023-01-02T00:00:00Z",
                "name": "John",
            }
        )
        assert params.eq_conditions == {
            "gmt_create": "2023-01-01T00:00:00Z",
            "gmt_modified": "2023-01-02T00:00:00Z",
            "name": "John",
        }

        # Test that invalid fields still raise errors
        with pytest.raises(ValueError) as exc_info:
            PageQueryParams[DummyModel](eq_conditions={"invalid_field": "value"})
        assert "Invalid fields found: {'invalid_field'}" in str(exc_info.value)

    def test_tag_filter_valid(self):
        """标签过滤器：合法字段"""
        from whiskerrag_types.model.page import Condition, Operator, TagFilter

        # 直接使用允许的字段
        tf = TagFilter(
            operator=Operator.AND,
            conditions=[
                Condition(field="tag_name", operator="eq", value="科技"),
                Condition(field="tag_id", operator="neq", value="123"),
            ],
        )
        assert isinstance(tf, TagFilter)
        assert tf.conditions[0].field == "tag_name"
        assert tf.conditions[1].field == "tag_id"

    def test_tag_filter_invalid_field(self):
        """标签过滤器：非法字段应报错"""
        import pytest

        from whiskerrag_types.model.page import Condition, Operator, TagFilter

        with pytest.raises(ValueError) as exc_info:
            TagFilter(
                operator=Operator.AND,
                conditions=[
                    Condition(field="not_allowed_field", operator="eq", value="x")
                ],
            )
        assert "only {'tag_name', 'tag_id'} are supported" in str(exc_info.value)

    def test_tag_filter_nested_groups(self):
        """标签过滤器：嵌套组的校验"""
        import pytest

        from whiskerrag_types.model.page import (
            Condition,
            FilterGroup,
            Operator,
            TagFilter,
        )

        nested = FilterGroup(
            operator=Operator.OR,
            conditions=[
                Condition(field="tag_name", operator="eq", value="科技"),
                Condition(field="invalid_field", operator="eq", value="bad"),
            ],
        )

        with pytest.raises(ValueError) as exc_info:
            TagFilter(operator=Operator.AND, conditions=[nested])
        assert "invalid_field" in str(exc_info.value)

    def test_query_params_with_tag_filter(self):
        """QueryParams 支持 TagFilter 并进行校验"""
        from whiskerrag_types.model.page import (
            Condition,
            Operator,
            QueryParams,
            TagFilter,
        )

        tf = TagFilter(
            operator=Operator.AND,
            conditions=[
                Condition(field="tag_name", operator="eq", value="科技"),
            ],
        )
        # 使用 DummyModel 作为泛型参数
        params = QueryParams[DummyModel](tag_filter=tf)
        assert isinstance(params.tag_filter, TagFilter)
        assert params.tag_filter.conditions[0].field == "tag_name"
