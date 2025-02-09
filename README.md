# WhiskerRAG-toolkit

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python Version](https://img.shields.io/pypi/pyversions/whiskerRAG)](https://pypi.org/project/whiskerRAG/)
[![PyPI version](https://badge.fury.io/py/whiskerRAG.svg)](https://badge.fury.io/py/whiskerRAG)

WhiskerRAG-toolkit 是为 PeterCat 和 Whisker 项目开发的 RAG（Retrieval-Augmented Generation）工具包，提供完整的 RAG 相关类型定义和方法实现。

## 特性
- 领域建模类型
- 插件接口描述
- Github、S3 数据源加载器
- OpenAI Emedding

## 安装

使用 pip 安装：

```bash
pip install whiskerRAG
```

## 快速开始

该工具包提供两个核心模块：whiskerrag_utils 和 whiskerrag_types

```
from whiskerrag_utils.github.fileLoader import GithubFileLoader
from whiskerrag_types.interface import DBPluginInterface
from whiskerrag_types.model import Knowledge, Task, Tenant, PageParams, PageResponse
```

## 开发指南

环境配置
本项目使用 Poetry 进行依赖管理。首先安装 Poetry：

```
pip install poetry
```

安装依赖

```
poetry install
```

运行测试

```
# 运行单元测试
poetry run pytest


# 生成测试覆盖率报告
poetry run pytest --cov

# 生成 HTML 格式的覆盖率报告
poetry run pytest --cov --cov-report=html
open htmlcov/index.html
```

## 构建与发布

```
# 构建项目
poetry build

# 发布到 PyPI
poetry publish
```
