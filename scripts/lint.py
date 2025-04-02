#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import logging


class Linter:

    def __init__(self) -> None:
        self.root_dir = Path(".")
        self.src_dirs = ["src", "tests"]
        self.setup_logging()

    def setup_logging(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.logger = logging.getLogger(__name__)

    def run_command(self, command: List[str], description: str) -> Tuple[bool, str]:
        self.logger.info(f"Running {description}...")
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                self.logger.error(f"{description} failed:")
                self.logger.error(result.stdout + result.stderr)
                return False, result.stdout + result.stderr
            return True, result.stdout
        except Exception as e:
            self.logger.error(f"{description} failed with error: {e}")
            return False, str(e)

    def run_flake8(self) -> bool:
        return self.run_command(["flake8", "src", "tests"], "flake8")[0]

    def run_black(self) -> bool:
        return self.run_command(["black", "--check", "src", "tests"], "black")[0]

    def run_isort(self) -> bool:
        return self.run_command(["isort", "--check-only", "src", "tests"], "isort")[0]

    def run_mypy(self) -> bool:
        return self.run_command(["mypy", "src"], "mypy")[0]

    def run_all(self, fix: bool = False) -> bool:
        self.logger.info("Running linting checks...")

        success = True

        # Flake8
        if not self.run_flake8():
            success = False

        # Black
        if fix:
            self.run_command(["black", "src", "tests"], "black (fixing)")
        elif not self.run_black():
            success = False

        # Isort
        if fix:
            self.run_command(["isort", "src", "tests"], "isort (fixing)")
        elif not self.run_isort():
            success = False

        # Mypy
        if not self.run_mypy():
            success = False

        if success:
            self.logger.info("All linting checks passed!")
        else:
            self.logger.error("Some linting checks failed!")
            if not fix:
                self.logger.info(
                    "Try running with --fix to automatically fix some issues"
                )

        return success


def main(fix: bool = False) -> None:
    linter = Linter()
    success = linter.run_all(fix=fix)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run linting checks.")
    parser.add_argument(
        "--fix", action="store_true", help="Attempt to automatically fix issues"
    )
    args = parser.parse_args()

    main(fix=args.fix)
