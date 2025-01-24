from typing import List, Optional, TypeVar, Generic, Dict, Any
from pydantic import BaseModel, Field

T = TypeVar("T")


class OrCondition:
    field: str
    operator: str  # eq, neq, gt, gte, lt, lte, like, ilike etc.
    value: Any


class PageParams(BaseModel):
    page: int = Field(default=1, ge=1, description="page number")
    page_size: int = Field(default=10, ge=1, le=100, description="page size")
    order_by: Optional[str] = Field(default=None, description="order by field")
    order_direction: Optional[str] = Field(default="asc", description="asc or desc")
    eq_conditions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="list of equality conditions, each as a dict with key and value",
    )

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        return self.page_size


class PageResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
