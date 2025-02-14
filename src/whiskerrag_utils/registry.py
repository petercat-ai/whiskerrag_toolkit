import importlib
import os
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Literal, Type, TypeVar, Union, overload

from whiskerrag_types.interface.embed_interface import BaseEmbedding
from whiskerrag_types.interface.loader_interface import BaseLoader
from whiskerrag_types.interface.retriever_interface import BaseRetriever
from whiskerrag_types.model.knowledge import (
    EmbeddingModelEnum,
    KnowledgeSourceEnum,
    KnowledgeTypeEnum,
)
from whiskerrag_types.model.retrieval import RetrievalEnum

RegisterKeyType = Union[
    KnowledgeSourceEnum, KnowledgeTypeEnum, EmbeddingModelEnum, RetrievalEnum
]
BaseRegisterClsType = Union[
    Type[BaseLoader], Type[BaseEmbedding], Type[BaseRetriever], None
]


class RegisterTypeEnum(str, Enum):
    EMBEDDING = "embedding"
    KNOWLEDGE_LOADER = "knowledge_loader"
    RETRIEVER = "retriever"


T_Embedding = TypeVar("T_Embedding", bound=BaseEmbedding)
T_Loader = TypeVar("T_Loader", bound=BaseLoader)
T_Retriever = TypeVar("T_Retriever", bound=BaseRetriever)

RegisteredType = Union[Type[T_Embedding], Type[T_Loader], Type[T_Retriever]]

RegisterDict = Dict[RegisterKeyType, RegisteredType]
_registry: Dict[RegisterTypeEnum, RegisterDict] = {
    RegisterTypeEnum.EMBEDDING: {},
    RegisterTypeEnum.KNOWLEDGE_LOADER: {},
    RegisterTypeEnum.RETRIEVER: {},
}
_loaded_packages = set()


def register(
    register_type: RegisterTypeEnum,
    register_key: RegisterKeyType,
) -> Callable[[RegisteredType], RegisteredType]:
    def decorator(cls: RegisteredType) -> RegisteredType:
        setattr(cls, "_is_register_item", True)
        setattr(cls, "_register_type", register_type)
        setattr(cls, "_register_key", register_key)
        expected_base: BaseRegisterClsType = None
        if register_type == RegisterTypeEnum.EMBEDDING:
            expected_base = BaseEmbedding
        elif register_type == RegisterTypeEnum.KNOWLEDGE_LOADER:
            expected_base = BaseLoader
        elif register_type == RegisterTypeEnum.RETRIEVER:
            expected_base = BaseRetriever
        else:
            raise ValueError(f"Unknown register type: {register_type}")

        if not issubclass(cls, expected_base):
            raise TypeError(
                f"Class {cls.__name__} must inherit from {expected_base.__name__}"
            )

        print(f"Registering {cls.__name__} as {register_type} with key {register_key}")
        _registry[register_type][register_key] = cls
        return cls

    return decorator


def init_register(package_name: str = "whiskerrag_utils") -> None:
    if package_name in _loaded_packages:
        return
    try:
        package = importlib.import_module(package_name)
        if package.__file__ is None:
            raise ValueError(
                f"Package {package_name} does not have a __file__ attribute"
            )
        package_path = Path(package.__file__).parent
        current_file = Path(__file__).name
        for root, _, files in os.walk(package_path):
            for file in files:
                if file == current_file or not file.endswith(".py"):
                    continue
                module_name = (
                    Path(root, file)
                    .relative_to(package_path)
                    .with_suffix("")
                    .as_posix()
                    .replace("/", ".")
                )

                if module_name == "__init__":
                    continue
                try:
                    importlib.import_module(f"{package_name}.{module_name}")
                except ImportError as e:
                    print(f"Error importing module {module_name}: {e}")
        _loaded_packages.add(package_name)

    except ImportError as e:
        print(f"Error importing package {package_name}: {e}")


@overload
def get_register(
    register_type: Literal[RegisterTypeEnum.KNOWLEDGE_LOADER],
    register_key: KnowledgeSourceEnum,
) -> Type[T_Loader]: ...


@overload
def get_register(
    register_type: Literal[RegisterTypeEnum.EMBEDDING], register_key: EmbeddingModelEnum
) -> Type[T_Embedding]: ...


@overload
def get_register(
    register_type: Literal[RegisterTypeEnum.RETRIEVER], register_key: RetrievalEnum
) -> Type[T_Retriever]: ...


def get_register(
    register_type: RegisterTypeEnum, register_key: RegisterKeyType
) -> RegisteredType:
    register_cls = _registry.get(register_type, {}).get(register_key, None)
    if register_cls is None:
        raise KeyError(f"No loader registered for type: {register_type}.{register_key}")
    return register_cls
