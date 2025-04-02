#!/usr/bin/env python3
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class HooksManager:
    def __init__(self) -> None:
        self.setup_logging()

    def setup_logging(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.logger = logging.getLogger(__name__)

    def run_command(self, command: List[str], description: str) -> Tuple[bool, str]:
        try:
            self.logger.info(f"Running: {description}")
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            if result.stdout:
                print(result.stdout)
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)
            return False, e.stderr

    def check_pre_commit_config(self) -> bool:
        config_file = Path(".pre-commit-config.yaml")
        if not config_file.exists():
            self.logger.error(".pre-commit-config.yaml not found")
            return False
        return True

    def install_hooks(self) -> bool:
        if not self.check_pre_commit_config():
            return False

        success, output = self.run_command(
            ["pre-commit", "install"], "Installing pre-commit hooks"
        )

        if success:
            self.logger.info("Pre-commit hooks installed successfully!")
        else:
            self.logger.error("Failed to install pre-commit hooks!")

        return success

    def run_hooks(self, all_files: bool = True) -> bool:
        if not self.check_pre_commit_config():
            return False

        command = ["pre-commit", "run"]
        if all_files:
            command.append("--all-files")

        success, output = self.run_command(command, "Running pre-commit hooks")

        if success:
            self.logger.info("All pre-commit hooks passed successfully!")
        else:
            self.logger.error("Some pre-commit hooks failed!")

        return success

    def setup_and_run(self, all_files: bool = True, install_only: bool = False) -> bool:
        if not self.install_hooks():
            return False

        if install_only:
            return True

        return self.run_hooks(all_files)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Manage pre-commit hooks")
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Only install the pre-commit hooks without running them",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        default=True,
        help="Run on all files (default: True)",
    )
    parser.add_argument(
        "--staged-only", action="store_true", help="Run only on staged files"
    )
    args = parser.parse_args()

    manager = HooksManager()
    success = manager.setup_and_run(
        all_files=not args.staged_only, install_only=args.install_only
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
