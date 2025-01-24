from abc import ABC, abstractmethod
from typing import TypeVar, Type, Dict, List
from pydantic import BaseModel

from .settings_interface import SettingsInterface
from .logger_interface import LoggerManagerInterface
from ..model import Task, Knowledge, Tenant, PageParams, PageResponse

T = TypeVar("T", bound=BaseModel)


class DBPluginInterface(ABC):
    settings: SettingsInterface
    logger: LoggerManagerInterface

    def __init__(self, logger: LoggerManagerInterface, settings: SettingsInterface):
        logger.info("DB plugin is initializing...")
        self.settings = settings
        self.logger = logger
        self.init()
        logger.info("DB plugin is initialized")

    @abstractmethod
    async def init(self):
        pass

    @abstractmethod
    async def get_db_client(self):
        pass

    @abstractmethod
    async def save_knowledge_list(
        self, knowledge_list: List[Knowledge]
    ) -> List[Knowledge]:
        pass

    @abstractmethod
    async def get_knowledge_list(
        self, space_id: str, page_params: PageParams
    ) -> PageResponse[Knowledge]:
        pass

    @abstractmethod
    async def get_knowledge(self, knowledge_id: str) -> Knowledge:
        pass

    @abstractmethod
    async def update_knowledge(self, knowledge: Knowledge):
        pass

    @abstractmethod
    async def delete_knowledge(self, knowledge_id_list: List[str]):
        pass

    @abstractmethod
    async def save_task_list(self, task_list: List[Task]):
        pass

    @abstractmethod
    async def get_tenant_by_id(self, tenant_id: str):
        pass

    @abstractmethod
    async def validate_tenant_by_sk(self, secret_key: str) -> bool:
        pass

    @abstractmethod
    async def get_tenant_by_sk(self, secret_key: str) -> Tenant | None:
        pass

    @abstractmethod
    def get_paginated_data(
        self,
        table_name: str,
        model_class: Type[T],
        page_params: PageParams,
    ) -> PageResponse[T]:
        pass
