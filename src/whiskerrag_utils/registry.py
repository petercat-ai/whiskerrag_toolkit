import importlib
import os
from pathlib import Path
from typing import Type, Union

from whiskerrag_types.interface.embed_interface import BaseEmbedding
from whiskerrag_types.interface.loader_interface import BaseLoader
from whiskerrag_types.model.knowledge import KnowledgeSourceType, KnowledgeType

RegisterType = Union[KnowledgeSourceType, KnowledgeType]
RegisterItem = Union[BaseLoader, BaseEmbedding]

_registry = {}
_loaded_packages = set()


def register(register_type: Union[KnowledgeSourceType, KnowledgeType]):
    def decorator(cls):
        cls._is_register_item = True
        if not issubclass(cls, (BaseLoader, BaseEmbedding)):
            raise TypeError(
                f"Class {cls.__name__} must inherit from BaseLoader or BaseEmbedding"
            )
        _registry[register_type] = cls
        return cls

    return decorator


def init_register(package_name: str = "whiskerrag_utils") -> None:
    if package_name in _loaded_packages:
        return
    try:
        package = importlib.import_module(package_name)
        package_path = Path(package.__file__).parent
        current_file = Path(__file__).name
        for root, _, files in os.walk(package_path):
            for file in files:
                # 跳过当前文件和非 Python 文件
                if file == current_file or not file.endswith(".py"):
                    continue
                # 构建模块路径
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
                    module = importlib.import_module(f"{package_name}.{module_name}")
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if getattr(attr, "_is_register_item", False):
                            _registry[attr_name] = attr

                except ImportError as e:
                    print(f"Error importing module {module_name}: {e}")

        _loaded_packages.add(package_name)

    except ImportError as e:
        print(f"Error importing package {package_name}: {e}")


def get_register(loader_type: RegisterType) -> Type[RegisterItem]:
    loader = _registry.get(loader_type)
    if loader is None:
        raise KeyError(f"No loader registered for type: {loader_type}")
    return loader
