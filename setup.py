import pathlib

from setuptools import find_namespace_packages, setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="whiskerrag",
    version="0.0.18",
    description="A utility package for RAG operations",
    long_description=README,
    long_description_content_type="text/markdown",
    author="petercat.ai",
    author_email="antd.antgroup@gmail.com",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8.1,<4.0",
    setup_requires=[
        "wheel",
        "setuptools>=42.0.0",
    ],
    install_requires=[
        "pydantic>=2.0.0,<3.0.0",
        "PyGithub==2.3.0",
        "langchain-community>=0.2.11",
        "langchain-openai>=0.1.20",
        "langchain-core>=0.2.28",
        "langchain>=0.2.12",
        "requests>=2.32.3",
        "typing-extensions>=4.12.2",
    ],
    url="https://github.com/petercat-ai/whiskerrag_toolkit",
    project_urls={
        "Bug Tracker": "https://github.com/petercat-ai/whiskerrag_toolkit/issues",
        "Documentation": "https://github.com/petercat-ai/whiskerrag_toolkit#README.md",
        "Source Code": "https://github.com/petercat-ai/whiskerrag_toolkit",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
