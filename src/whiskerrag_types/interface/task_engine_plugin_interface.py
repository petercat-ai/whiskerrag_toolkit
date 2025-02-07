from abc import ABC, abstractmethod
from typing import List

from ..interface import SettingsInterface, LoggerManagerInterface, DBPluginInterface
from ..model import Tenant, KnowledgeCreate, Knowledge, Task


class TaskEnginPluginInterface(ABC):
    settings: SettingsInterface
    logger: LoggerManagerInterface

    def __init__(
        self,
        logger: LoggerManagerInterface,
        settings: SettingsInterface,
    ):
        try:
            logger.info("TaskEngine plugin is initializing...")
            self.settings = settings
            self.logger = logger
            self.init()
            logger.info("TaskEngine plugin is initialized")
        except Exception as e:
            logger.error(f"TaskEngine plugin init error: {e}")

    @abstractmethod
    def init(self):
        """
        Initialize the task engine plugin, such as loading middleware, establishing contact with the task execution engine, etc.
        """
        pass

    @abstractmethod
    async def gen_knowledge_list(
        self, user_input: List[KnowledgeCreate], tenant: Tenant
    ) -> List[Knowledge]:
        """
        Generate a list of tasks from the user input.
        """
        pass

    @abstractmethod
    async def init_task_from_knowledge(
        self, knowledge_list: List[Knowledge], tenant: Tenant
    ) -> List[Task]:
        """
        Initialize a list of tasks from the knowledge list.
        """
        pass

    @abstractmethod
    async def execute_task(self, task_id: str) -> List[Task]:
        """
        Execute a task.
        """
        pass

    @abstractmethod
    async def batch_execute_task(
        self, task_list: List[Task], knowledge_list: List[Knowledge]
    ) -> List[Task]:
        """
        Execute a list of tasks.
        """
        pass

    @abstractmethod
    async def on_task_execute(self, db: DBPluginInterface):
        """
        Listen to the task execution status with no parameter restrictions.
        """
        pass
