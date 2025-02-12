# WhiskerRAG

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python Version](https://img.shields.io/pypi/pyversions/whiskerrag)](https://pypi.org/project/whiskerrag/)
[![PyPI version](https://badge.fury.io/py/whiskerrag.svg)](https://badge.fury.io/py/whiskerrag)

WhiskerRAG 是为 PeterCat 和 Whisker 项目开发的 RAG（Retrieval-Augmented Generation）工具包，提供完整的 RAG 相关类型定义和方法实现。

## 特性

- 领域建模类型
- 插件接口描述
- Github、S3 数据源加载器
- OpenAI Emedding

## 安装

使用 pip 安装：

```bash
pip install whiskerrag
```

## 快速开始

该工具包提供两个核心模块：whiskerrag_utils 和 whiskerrag_types

```python
from whiskerrag_utils.github.fileLoader import GithubFileLoader
from whiskerrag_types.interface import DBPluginInterface
from whiskerrag_types.model import Knowledge, Task, Tenant, PageParams, PageResponse
```

## 开发指南

### 环境设置

1. 克隆项目

```bash
git clone https://github.com/your-username/whiskerRAG-toolkit.git
cd whiskerRAG-toolkit
```

2. 创建并激活虚拟环境

```bash
make setup
source venv/bin/activate
```

### 开发工作流

1. 代码格式化

```bash
make format
```

2. 运行测试

```bash
# 运行所有测试
make test

# 运行特定测试文件
make test-file file=tests/test_specific.py
```

3. 代码检查

```bash
# 运行所有检查（lint, type check, test）
make check

# 仅运行 lint
make lint

# 仅运行类型检查
make lint-mypy
```

4. 生成测试覆盖率报告

```bash
make coverage
```

### 分支管理

创建新的功能分支：

```bash
make branch name=feature/new-feature
```

### 构建和发布

1. 构建包

```bash
make build
```

2. 检查构建的包

```bash
make check-build
```

3. 发布到 TestPyPI

```bash
make upload-test
```

4. 发布到 PyPI

```bash
make upload
```

5. 创建新版本发布

```bash
make release new_version=X.Y.Z
```

### 其他命令

- 清理构建文件和缓存：

```bash
make clean
```

- 更新依赖版本：

```bash
make update-deps
```

- 运行 pre-commit 钩子：

```bash
make pre-commit
```

## 项目结构

```
whiskerRAG-toolkit/
├── src/
│   ├── whiskerrag_utils/
│   └── whiskerrag_types/
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── Makefile
```

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`make branch name=feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

项目维护者 - [@petercat-ai](https://github.com/petercat-ai)

项目链接：[https://github.com/petercat-ai/whiskerrag_toolkit](https://github.com/your-username/whiskerrag_toolkit)
