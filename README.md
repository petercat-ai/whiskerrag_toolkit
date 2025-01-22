# WhiskerRAG-toolkit
petercat、whisker 项目使用的 rag 工具包，提供 rag 相关类型定义和方法

## 使用方式
```
from whisker_rag_util.github.fileLoader import GithubFileLoader
...
```

## 项目启动
安装 poetry 进行依赖管理

```bash
pip install poetry
```

项目根目录下安装依赖

```bash
poetry install
```

## 本地测试

```bash
# 运行测试
poetry run pytest

# 带覆盖率报告
poetry run pytest --cov

# 查看HTML格式的覆盖率报告
poetry run pytest --cov --cov-report=html
open htmlcov/index.html

```

# 构建并发布

```bash
poetry build

poetry publish
```
