from abc import ABC, abstractmethod

from whiskerrag_types.model.knowledge import Knowledge


class BaseLoader(ABC):
    def __init__(self, knowledge: Knowledge):
        self.knowledge = knowledge

    @abstractmethod
    async def load(self) -> str:
        pass
