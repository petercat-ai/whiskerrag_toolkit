from abc import ABC, abstractmethod
from typing import List

from ..interface import LoggerManagerInterface, SettingsInterface
from ..model import Knowledge, Task, Tenant


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
    def init(self) -> None:
        """
        Initialize the task engine plugin, such as loading middleware, establishing contact with the task execution engine, etc.
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
    async def batch_execute_task(
        self, task_list: List[Task], knowledge_list: List[Knowledge]
    ) -> List[Task]:
        """
        Execute a list of tasks.
        """
        pass
