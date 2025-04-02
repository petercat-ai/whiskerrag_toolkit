#!/usr/bin/env python3
from enum import Enum
import subprocess
import sys
from pathlib import Path
import re
from typing import Optional, Tuple
import logging
from packaging import version


class ReleaseType(Enum):
    ALPHA = "alpha"
    BETA = "beta"
    RC = "rc"
    FINAL = "final"


class ReleaseManager:
    def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.logger = logging.getLogger(__name__)

    def run_command(self, command: list, description: str) -> bool:
        """运行命令并返回是否成功"""
        try:
            self.logger.info(f"Running: {description}")
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.logger.info(result.stdout.strip())
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error in {description}: {e.stderr.strip()}")
            return False

    def get_current_version(self) -> str:
        """获取当前版本号"""
        try:
            result = subprocess.run(
                ["poetry", "version", "-s"], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error getting current version: {e}")
            sys.exit(1)

    def validate_version(self, version_str: str) -> bool:
        """验证版本号格式"""
        try:
            self.logger.info(f"version_str:{version_str}")
            v = version.parse(version_str)
            # 检查是否符合期望的格式
            valid_pre = ["a", "b", "rc"]  # alpha, beta, rc
            if v.pre and v.pre[0] not in valid_pre:
                self.logger.error(
                    f"Invalid pre-release identifier. Use 'a' for alpha, 'b' for beta, or 'rc' for release candidate"
                )
                return False
            return True
        except version.InvalidVersion:
            self.logger.error(f"Invalid version format: {version_str}")
            self.logger.info("Valid formats examples:")
            self.logger.info("  - Release versions: 1.0.0, 2.1.0, 1.0.1")
            self.logger.info("  - Alpha versions: 1.0.0a1, 1.0.0a2")
            self.logger.info("  - Beta versions: 1.0.0b1, 1.0.0b2")
            self.logger.info("  - Release candidates: 1.0.0rc1, 1.0.0rc2")
            return False

    def update_version(self, new_version: str) -> bool:
        """更新版本号"""
        return self.run_command(
            ["poetry", "version", new_version], f"Updating version to {new_version}"
        )

    def commit_and_tag(self, new_version: str) -> bool:
        """提交更改并创建标签"""
        commands = [
            (["git", "add", "pyproject.toml"], "Adding pyproject.toml"),
            (
                ["git", "commit", "-m", f"Release version {new_version}"],
                "Committing changes",
            ),
            (
                ["git", "tag", "-a", f"v{new_version}", "-m", f"Version {new_version}"],
                "Creating tag",
            ),
            (["git", "push", "origin", f"v{new_version}"], "Pushing tag"),
        ]

        try:
            for cmd, desc in commands:
                if not self.run_command(cmd, desc):
                    # 回滚操作
                    if desc != "Adding pyproject.toml":
                        self.run_command(
                            ["git", "reset", "--hard", "HEAD~1"], "Rolling back commit"
                        )
                        self.run_command(
                            ["git", "tag", "-d", f"v{new_version}"], "Removing tag"
                        )
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error during git operations: {e}")
            return False

    def build_and_publish(self) -> bool:
        """构建并发布包"""
        commands = [
            (["poetry", "build"], "Building package"),
            (["poetry", "publish"], "Publishing package"),
        ]

        for cmd, desc in commands:
            if not self.run_command(cmd, desc):
                return False
        return True

    def parse_version_type(
        self, version_str: str
    ) -> Tuple[version.Version, ReleaseType]:
        v = version.parse(version_str)
        if v.is_prerelease:
            pre = v.pre[0] if v.pre else ""
            if "a" in pre:
                return v, ReleaseType.ALPHA
            elif "b" in pre:
                return v, ReleaseType.BETA
            elif "rc" in pre:
                return v, ReleaseType.RC
        return v, ReleaseType.FINAL

    def validate_version_sequence(self, current: str, new: str) -> bool:
        current_v, current_type = self.parse_version_type(current)
        new_v, new_type = self.parse_version_type(new)

        if current_v.release == new_v.release and new_v.is_prerelease:
            return True

        # 如果新版本是预发布版本，但主版本号更大，允许
        if new_v.is_prerelease and new_v.release > current_v.release:
            return True

        # 如果当前是预发布版本，新版本是相同主版本的正式版本，允许
        if (
            current_v.is_prerelease
            and not new_v.is_prerelease
            and current_v.release == new_v.release
        ):
            return True

        # 其他情况下，新版本必须大于当前版本
        return new_v > current_v

    def release(self, new_version: Optional[str] = None) -> bool:
        """执行发布流程"""
        current_version = self.get_current_version()
        self.logger.info(f"Current version: {current_version}")

        if new_version is None:
            self.logger.info("\nVersion format examples:")
            self.logger.info("  - Release versions: 1.0.0, 2.1.0, 1.0.1")
            self.logger.info("  - Alpha versions: 1.0.0a1, 1.0.0a2")
            self.logger.info("  - Beta versions: 1.0.0b1, 1.0.0b2")
            self.logger.info("  - Release candidates: 1.0.0rc1, 1.0.0rc2")
            new_version = input(
                f"\nEnter new version (current: {current_version}): "
            ).strip()

        if not new_version:
            self.logger.error("Version number is required")
            return False

        if not self.validate_version(new_version):
            return False

        if not self.validate_version_sequence(current_version, new_version):
            self.logger.error(
                f"Invalid version sequence: {current_version} -> {new_version}\n"
                "Version sequence must follow these rules:\n"
                "1. Pre-release versions (alpha/beta/rc) can be lower than current version if same major version\n"
                "2. New pre-release version must have higher major version than current release version\n"
                "3. Final release version must be higher than its pre-release versions\n"
                "4. Final release version must be higher than previous final release"
            )
            return False

        # 添加确认步骤
        confirm = input(f"Ready to release version {new_version}? [y/N] ").lower()
        if confirm != "y":
            self.logger.info("Release cancelled")
            return False

        # 执行发布流程
        if not self.update_version(new_version):
            return False

        if not self.commit_and_tag(new_version):
            return False

        if not self.build_and_publish():
            return False

        self.logger.info(f"Successfully released version {new_version}")
        return True

    def check_environment(self) -> bool:
        try:
            # 检查 git 仓库状态
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout.strip():
                self.logger.error(
                    "Working directory is not clean. Please commit or stash changes first."
                )
                return False

            # 检查是否已登录到 PyPI
            if not self.run_command(
                ["poetry", "config", "pypi-token.pypi"], "Checking PyPI authentication"
            ):
                self.logger.error(
                    "Not authenticated with PyPI. Please configure PyPI token first."
                )
                return False

            return True
        except Exception as e:
            self.logger.error(f"Environment check failed: {e}")
            return False


def main() -> None:
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Release a new version")
    parser.add_argument("version", nargs="?", help="New version number (optional)")
    args = parser.parse_args()

    manager = ReleaseManager()
    success = manager.release(args.version)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
