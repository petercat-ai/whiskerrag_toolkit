from whiskerrag_types.model.knowledge import Knowledge, KnowledgeTypeEnum
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_types.model.splitter import MarkdownSplitConfig
from whiskerrag_utils import RegisterTypeEnum, get_register
from whiskerrag_utils.registry import init_register

markdown_content = """
这是第一个段落
# 一级标题
## 二级标题
### 三级标题

这是第一个段落。这里包含了中文句号。这里有英文句号.还有感叹号！以及问号？
下面是带着逗号的句子，还有中文逗号，都在这里了。

这是第二个段落，包含**粗体**和*斜体*文本。
这段文字包含`代码`格式。

下面是一个列表：
* 无序列表项1
* 无序列表项2
  * 缩进的子项
  * 另一个子项
* 无序列表项3

有序列表：
1. 第一项
2. 第二项
3. 第三项

使用减号的列表：
- 项目1
- 项目2
- 项目3

使用加号的列表：
+ 第一点
+ 第二点
+ 第三点

下面是一条分隔线：
***

再来一条分隔线：
---

> 这是一段引用文字。
> 引用可以有多行。

这里是一个表格：
| 列1 | 列2 | 列3 |
|-----|-----|-----|
| 内容1 | 内容2 | 内容3 |

下面是一个代码块：
```python
def hello():
    print("Hello, World!")
"""

knowledge_data = {
    "source_type": "user_input_text",
    "knowledge_type": KnowledgeTypeEnum.MARKDOWN,
    "space_id": "local_test",
    "knowledge_name": "local_test_5",
    "split_config": {},
    "source_config": {"text": markdown_content},
    "embedding_model_name": "openai",
    "tenant_id": "38fbd78b-1869-482c-9142-e43a2c2s6e42",
    "metadata": {},
}


class TestMarkdownSplitter:
    def test_split(self) -> None:
        knowledge = Knowledge(**knowledge_data)
        split_config = MarkdownSplitConfig(
            type="markdown",
            chunk_size=100,
            chunk_overlap=0,
            separators=["\n\n", " ", ""],
            keep_separator=False,
            is_separator_regex=False,
        )
        knowledge.update(split_config=split_config)
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.Parser, "markdown")
        res = SplitterCls().parse(
            knowledge, Text(content=markdown_content, metadata={})
        )
        assert len(res) == 6

    def test_split_extract_header(self) -> None:
        knowledge = Knowledge(**knowledge_data)
        split_config = MarkdownSplitConfig(
            type="markdown",
            chunk_size=100,
            chunk_overlap=0,
            separators=["\n\n", " ", ""],
            keep_separator=False,
            is_separator_regex=False,
            extract_header_first=True,
        )
        knowledge.update(split_config=split_config)
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.Parser, "markdown")
        res = SplitterCls().parse(
            knowledge, Text(content=markdown_content, metadata={})
        )
        assert len(res) == 9
