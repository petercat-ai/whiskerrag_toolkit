from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

from whiskerrag_types.model.utils import parse_datetime


class TaskRestartRequest(BaseModel):
    task_id_list: List[str] = Field(..., description="List of task IDs to restart")


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELED = "canceled"
    PENDING_RETRY = "pending_retry"


class Task(BaseModel):
    task_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="任务唯一标识符",
        alias="task_id",
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING, description="任务当前状态", alias="status"
    )
    knowledge_id: str = Field(..., description="文件来源标识符", alias="knowledge_id")
    metadata: Optional[dict] = Field(None, description="任务元数据", alias="metadata")
    error_message: Optional[str] = Field(
        None, description="错误信息（仅失败时存在）", alias="error_message"
    )
    space_id: str = Field(..., description="空间标识符", alias="space_id")
    user_id: Optional[str] = Field(None, description="用户标识符", alias="user_id")
    tenant_id: str = Field(..., description="租户标识符", alias="tenant_id")

    created_at: Optional[datetime] = Field(
        default=None,
        description="任务创建时间",
        alias="gmt_create",
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="最后更新时间",
        alias="gmt_modified",
    )

    def update(self, **kwargs: Any) -> "Task":
        if "created_at" in kwargs:
            raise ValueError("created_at 是不可修改的只读字段")
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.updated_at = datetime.now(timezone.utc)
        return self

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
    def set_defaults(self) -> "Task":
        now = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now
        return self

    class Config:
        extra = "ignore"
        allow_population_by_field_name = True
        serialize_by_alias = False
