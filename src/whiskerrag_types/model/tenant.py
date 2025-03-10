from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_serializer, field_validator


class Tenant(BaseModel):
    tenant_id: str = Field(
        default_factory=lambda: str(uuid4()), description="tenant id"
    )
    tenant_name: str = Field("", description="tenant name")
    email: str = Field(..., description="email")
    secret_key: str = Field("", description="secret_key")
    is_active: bool = Field(True, description="is active")
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(),
        alias="gmt_create",
        description="creation time",
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(),
        alias="gmt_modified",
        description="update time",
    )

    @field_serializer("created_at")
    def serialize_created_at(self, created_at: Optional[datetime]) -> Optional[str]:
        return created_at.isoformat() if created_at else None

    @field_serializer("updated_at")
    def serialize_updated_at(self, updated_at: Optional[datetime]) -> Optional[str]:
        return updated_at.isoformat() if updated_at else None

    def update(self, **kwargs: dict) -> "Tenant":
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.updated_at = datetime.now()
        return self

    @field_validator("is_active", mode="before")
    @classmethod
    def convert_tinyint_to_bool(cls, v: Any) -> bool:
        return bool(v)

    class Config:
        allow_population_by_field_name = True
