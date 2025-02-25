from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_serializer


class Tenant(BaseModel):
    tenant_id: str = Field(
        default_factory=lambda: str(uuid4()), description="tenant id"
    )
    tenant_name: str = Field("", description="tenant name")
    email: str = Field(..., description="email")
    secret_key: str = Field("", description="secret_key")
    is_active: bool = Field(True, description="is active")
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(), description="creation time"
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(), description="update time"
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
