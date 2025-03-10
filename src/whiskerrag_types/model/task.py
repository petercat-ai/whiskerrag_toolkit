from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_serializer


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
    """
    Task model representing a task entity with various attributes.
    Attributes:
        task_id (str): Task ID.
        status (TaskStatus): Status of the task.
        knowledge_id (str): File source information.
        space_id (str): Space ID.
        user_id (Optional[str]): User ID.
        tenant_id (str): Tenant ID.
        created_at (Optional[datetime]): Creation time, defaults to current time.
        updated_at (Optional[datetime]): Update time, defaults to current time.
    Methods:
        serialize_created_at(created_at: Optional[datetime]) -> str:
            Serializes the created_at attribute to ISO format.
        serialize_updated_at(updated_at: Optional[datetime]) -> str:
            Serializes the updated_at attribute to ISO format.
        update(**kwargs) -> Task:
            Updates the task attributes with provided keyword arguments and sets updated_at to current time.
    """

    task_id: str = Field(default_factory=lambda: str(uuid4()), description="task id")
    status: TaskStatus = Field(TaskStatus.PENDING, description="task status")
    knowledge_id: str = Field(..., description="file source info")
    metadata: Optional[dict] = Field(
        None, description="task metadata info, make task readable"
    )
    error_message: Optional[str] = Field(None, description="error message")
    space_id: str = Field(..., description="space id")
    user_id: Optional[str] = Field(None, description="user id")
    tenant_id: str = Field(..., description="tenant id")
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

    @field_serializer("status")
    def serialize_status(self, status: TaskStatus) -> str:
        return status.value if isinstance(status, TaskStatus) else str(status)

    @field_serializer("created_at")
    def serialize_created_at(self, created_at: Optional[datetime]) -> Optional[str]:
        return created_at.isoformat() if created_at else None

    @field_serializer("updated_at")
    def serialize_updated_at(self, updated_at: Optional[datetime]) -> Optional[str]:
        return updated_at.isoformat() if updated_at else None

    def update(self, **kwargs: dict) -> "Task":
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.updated_at = datetime.now()
        return self

    class Config:
        extra = "ignore"
        allow_population_by_field_name = True
