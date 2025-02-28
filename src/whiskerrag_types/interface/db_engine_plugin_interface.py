from abc import ABC, abstractmethod
from typing import Any, List, TypeVar, Union

from pydantic import BaseModel

from whiskerrag_types.model.chunk import Chunk
from whiskerrag_types.model.retrieval import (
    RetrievalByKnowledgeRequest,
    RetrievalBySpaceRequest,
    RetrievalChunk,
)

from ..model import Knowledge, PageParams, PageResponse, Task, Tenant
from .logger_interface import LoggerManagerInterface
from .settings_interface import SettingsInterface

T = TypeVar("T", bound=BaseModel)


class DBPluginInterface(ABC):
    settings: SettingsInterface
    logger: LoggerManagerInterface

    def __init__(
        self, logger: LoggerManagerInterface, settings: SettingsInterface
    ) -> None:
        logger.info("DB plugin is initializing...")
        self.settings = settings
        self.logger = logger
        self.init()
        logger.info("DB plugin is initialized")

    @abstractmethod
    def init(self) -> None:
        pass

    @abstractmethod
    def get_db_client(self) -> Any:
        pass

    # =================== knowledge ===================
    @abstractmethod
    async def save_knowledge_list(
        self, knowledge_list: List[Knowledge]
    ) -> List[Knowledge]:
        pass

    @abstractmethod
    async def get_knowledge_list(
        self, tenant_id: str, page_params: PageParams[Knowledge]
    ) -> PageResponse[Knowledge]:
        pass

    @abstractmethod
    async def get_knowledge(self, tenant_id: str, knowledge_id: str) -> Knowledge:
        pass

    @abstractmethod
    async def update_knowledge(self, knowledge: Knowledge) -> None:
        pass

    @abstractmethod
    async def delete_knowledge(
        self, tenant_id: str, knowledge_id_list: List[str]
    ) -> None:
        pass

    # =================== chunk ===================
    @abstractmethod
    async def save_chunk_list(self, chunks: List[Chunk]) -> List[Chunk]:
        pass

    @abstractmethod
    async def get_chunk_list(
        self, tenant_id: str, page_params: PageParams[Chunk]
    ) -> List[Chunk]:
        pass

    @abstractmethod
    async def get_chunk_by_id(self, tenant_id: str, chunk_id: str) -> Chunk:
        pass

    @abstractmethod
    async def delete_knowledge_chunk(
        self, tenant_id: str, knowledge_ids: List[str]
    ) -> Union[List[Chunk], None]:
        pass

    # =================== retrieval ===================
    @abstractmethod
    async def search_space_chunk_list(
        self,
        tenant_id: str,
        params: RetrievalBySpaceRequest,
    ) -> List[RetrievalChunk]:
        pass

    @abstractmethod
    async def search_knowledge_chunk_list(
        self,
        tenant_id: str,
        params: RetrievalByKnowledgeRequest,
    ) -> List[RetrievalChunk]:
        pass

    # =================== task ===================
    @abstractmethod
    async def save_task_list(self, task_list: List[Task]) -> List[Task]:
        pass

    @abstractmethod
    async def update_task_list(self, task_list: List[Task]) -> None:
        pass

    @abstractmethod
    async def get_task_list(
        self, tenant_id: str, page_params: PageParams[Task]
    ) -> PageResponse[Task]:
        pass

    @abstractmethod
    async def get_task_by_id(self, tenant_id: str, task_id: str) -> Union[Task, None]:
        pass

    @abstractmethod
    async def delete_knowledge_task(
        self, tenant_id: str, knowledge_ids: List[str]
    ) -> Union[List[Task], None]:
        pass

    # =================== tenant ===================
    @abstractmethod
    async def save_tenant(self, tenant: Tenant) -> Union[Tenant, None]:
        pass

    @abstractmethod
    async def get_tenant_by_sk(self, secret_key: str) -> Union[Tenant, None]:
        pass

    @abstractmethod
    async def update_tenant(self, tenant: Tenant) -> None:
        pass

    @abstractmethod
    async def validate_tenant_name(self, tenant_name: str) -> bool:
        pass
