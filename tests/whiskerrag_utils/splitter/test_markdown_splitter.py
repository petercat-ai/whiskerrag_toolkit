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


class TestMarkdownSplitter:
    def test_split(self) -> None:
        split_config = MarkdownSplitConfig(
            type="markdown",
            chunk_size=100,
            chunk_overlap=0,
            separators=["\n\n", " ", ""],
            keep_separator=False,
            is_separator_regex=False,
        )
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.SPLITTER, "markdown")
        res = SplitterCls().split(
            Text(content=markdown_content, metadata={}), split_config
        )
        assert len(res) == 6

    def test_split_extract_header(self) -> None:
        split_config = MarkdownSplitConfig(
            type="markdown",
            chunk_size=100,
            chunk_overlap=0,
            separators=["\n\n", " ", ""],
            keep_separator=False,
            is_separator_regex=False,
            extract_header_first=True,
        )
        init_register()
        SplitterCls = get_register(RegisterTypeEnum.SPLITTER, "markdown")
        res = SplitterCls().split(
            Text(content=markdown_content, metadata={}), split_config
        )
        assert len(res) == 9
