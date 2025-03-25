from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from whiskerrag_types.model.utils import parse_datetime


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

    def update(self, **kwargs: dict) -> "Tenant":
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.updated_at = datetime.now()
        return self

    @field_validator("is_active", mode="before")
    @classmethod
    def convert_tinyint_to_bool(cls, v: Any) -> bool:
        return bool(v)

    @model_validator(mode="before")
    def ensure_aware_timezones(cls, values: dict) -> dict:
        field_mappings = {"created_at": "gmt_create", "updated_at": "gmt_modified"}
        for field, alias_name in field_mappings.items():
            val = values.get(field) or values.get(alias_name)
            if val is None:
                continue

            if isinstance(val, str):
                dt = parse_datetime(val)
                values[field] = dt
                values[alias_name] = dt
            else:
                if val and val.tzinfo is None:
                    dt = val.replace(tzinfo=timezone.utc)
                    values[field] = dt
                    values[alias_name] = dt

        return values

    @model_validator(mode="after")
    def set_defaults(self) -> "Tenant":
        now = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now
        return self

    class Config:
        allow_population_by_field_name = True
