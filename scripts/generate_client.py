# type: ignore
import os
import shutil
import tempfile
import subprocess
import platform
from pathlib import Path


def merge_requirements(generated_req_path, root_req_path):
    generated_requirements = set()
    if Path(generated_req_path).exists():
        with open(generated_req_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    req = line.split("#")[0].strip()
                    package_name = (
                        req.split(">=")[0]
                        .split("<=")[0]
                        .split("==")[0]
                        .split(">")[0]
                        .split("<")[0]
                        .strip()
                    )
                    generated_requirements.add((package_name, req))

    existing_requirements = set()
    if Path(root_req_path).exists():
        with open(root_req_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    req = line.split("#")[0].strip()
                    package_name = (
                        req.split(">=")[0]
                        .split("<=")[0]
                        .split("==")[0]
                        .split(">")[0]
                        .split("<")[0]
                        .strip()
                    )
                    existing_requirements.add((package_name, req))

    final_requirements = existing_requirements.copy()
    for gen_pkg, gen_req in generated_requirements:
        if not any(
            gen_pkg == existing_pkg for existing_pkg, _ in existing_requirements
        ):
            final_requirements.add((gen_pkg, gen_req))

    with open(root_req_path, "w") as f:
        f.write("# Generated and merged requirements\n")
        for _, req in sorted(final_requirements, key=lambda x: x[0].lower()):
            f.write(f"{req}\n")


def run_command(command) -> None:
    try:
        result = subprocess.run(
            command,
            shell=True if platform.system() == "Windows" else False,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error output: {e.stderr}")
        raise


def generate_client(openapi_path: str) -> None:
    tmp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {tmp_dir}")

    try:
        Path("src/whiskerrag_client").mkdir(parents=True, exist_ok=True)
        Path("tests/whiskerrag_client").mkdir(parents=True, exist_ok=True)

        command = [
            "openapi-generator-cli",
            "generate",
            "-i",
            openapi_path,
            "-g",
            "python",
            "-o",
            tmp_dir,
            "--additional-properties=packageName=whiskerrag_client",
        ]

        if platform.system() == "Windows":
            command = " ".join(command)

        print("Generating client code...")
        run_command(command)

        src_client_dir = Path(tmp_dir) / "whiskerrag_client"
        dest_client_dir = Path("src/whiskerrag_client")

        if src_client_dir.exists():
            if dest_client_dir.exists():
                shutil.rmtree(dest_client_dir)
            shutil.copytree(src_client_dir, dest_client_dir)
            print("Copied client code to src/whiskerrag_client/")

        src_test_dir = Path(tmp_dir) / "test"
        if src_test_dir.exists():
            for test_file in src_test_dir.glob("*"):
                shutil.copy2(test_file, "tests/whiskerrag_client/")
            print("Copied test files to tests/whiskerrag_client")

        generated_req = Path(tmp_dir) / "requirements.txt"
        root_req = Path("requirements.txt")
        if generated_req.exists():
            merge_requirements(generated_req, root_req)
            print("Merged requirements.txt")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        try:
            shutil.rmtree(tmp_dir)
            print(f"Cleaned up temporary directory: {tmp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")


if __name__ == "__main__":
    print("Starting client generation...")
    generate_client("http://127.0.0.1:8000/openapi.json")
    print("Client generation completed!")
