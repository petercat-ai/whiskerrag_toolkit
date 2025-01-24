from dataclasses import Field
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, field_serializer, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


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
    
    task_id: str = Field(None, description="task id")
    status: TaskStatus = Field(None, description="task status")
    knowledge_id: str = Field(None, description="file source info")
    space_id: str = Field(..., description="space id")
    user_id: Optional[str] = Field(None, description="user id")
    tenant_id: str = Field(..., description="tenant id")
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(), description="creation time"
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(), description="update time"
    )
    
    @field_serializer("status")
    def serialize_status(self, status: TaskStatus):
        return status.value if isinstance(status, TaskStatus) else str(status)

    @field_serializer("created_at")
    def serialize_created_at(self, created_at: Optional[datetime]):
        return created_at.isoformat() if created_at else None

    @field_serializer("updated_at")
    def serialize_updated_at(self, updated_at: Optional[datetime]):
        return updated_at.isoformat() if updated_at else None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.updated_at = datetime.now().isoformat()
        return self
