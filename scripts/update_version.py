import re
import sys


def update_version(new_version: str) -> None:
    with open("setup.py", "r") as f:
        content = f.read()

    content = re.sub(r'version="[^"]*"', f'version="{new_version}"', content)

    with open("setup.py", "w") as f:
        f.write(content)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_version.py VERSION")
        sys.exit(1)
    update_version(sys.argv[1])
