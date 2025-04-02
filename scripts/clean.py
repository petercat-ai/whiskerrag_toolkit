#!/usr/bin/env python3
import shutil
import os
from pathlib import Path


def clean() -> None:
    """Clean up build artifacts and cache directories."""
    print("Cleaning up...")

    # 要删除的目录列表
    dirs_to_remove = [
        "build",
        "dist",
        "*.egg-info",
        ".coverage",
        "htmlcov",
        ".pytest_cache",
        ".mypy_cache",
        "__pycache__",
    ]

    # 要删除的文件模式
    file_patterns = [
        "*.pyc",
    ]

    root_dir = Path(".")

    # 删除指定目录
    for dir_pattern in dirs_to_remove:
        for path in root_dir.glob(f"**/{dir_pattern}"):
            if path.is_dir():
                try:
                    shutil.rmtree(path)
                    print(f"Removed directory: {path}")
                except Exception as e:
                    print(f"Failed to remove {path}: {e}")

    # 删除指定文件
    for file_pattern in file_patterns:
        for path in root_dir.glob(f"**/{file_pattern}"):
            if path.is_file():
                try:
                    path.unlink()
                    print(f"Removed file: {path}")
                except Exception as e:
                    print(f"Failed to remove {path}: {e}")

    print("Clean complete.")


if __name__ == "__main__":
    clean()
