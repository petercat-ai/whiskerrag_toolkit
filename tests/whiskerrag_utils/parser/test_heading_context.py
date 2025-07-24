#!/usr/bin/env python3
"""
测试 YuqueParser 的标题上下文功能
"""
import asyncio
import os
import re
import sys
from typing import Dict, List, Tuple, Union

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(project_root, "src"))


# 简化的类定义，避免复杂的依赖
class Text:
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}


class Image:
    def __init__(self, url: str, metadata: dict = None):
        self.url = url
        self.metadata = metadata or {}


class YuqueSplitConfig:
    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=200,
        separators=None,
        is_separator_regex=True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.is_separator_regex = is_separator_regex


# 简化的 RecursiveCharacterTextSplitter
class RecursiveCharacterTextSplitter:
    def __init__(
        self, chunk_size=1000, chunk_overlap=200, separators=None, keep_separator=False
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.keep_separator = keep_separator

    def split_text(self, text: str) -> List[str]:
        # 简化的分割逻辑
        chunks = []
        current_chunk = ""

        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                for i, part in enumerate(parts):
                    if len(current_chunk) + len(part) <= self.chunk_size:
                        current_chunk += part
                        if i < len(parts) - 1:
                            current_chunk += separator
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = part
                        if i < len(parts) - 1:
                            current_chunk += separator
                break

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]


# 简化的 YuqueParser
class YuqueParser:
    def _extract_headings(self, content: str) -> List[Tuple[int, str, int]]:
        """
        Extract headings from markdown content.
        Returns list of (level, title, position) tuples.
        """
        heading_pattern = r"^(#{1,6})\s+(.+)$"
        headings = []

        for match in re.finditer(heading_pattern, content, re.MULTILINE):
            level = len(match.group(1))  # number of # characters
            title = match.group(2).strip()
            position = match.start()
            headings.append((level, title, position))

        return headings

    def _build_heading_hierarchy(
        self, headings: List[Tuple[int, str, int]]
    ) -> Dict[int, List[str]]:
        """
        Build heading hierarchy for heading positions only.
        Returns a mapping from heading position to list of hierarchical headings.
        Optimized to store only heading positions instead of every character position.
        """
        hierarchy_map = {}
        current_hierarchy: List[Union[str, None]] = [
            None
        ] * 6  # Support up to 6 levels of headings

        for level, title, position in headings:
            # Update current hierarchy
            current_hierarchy[level - 1] = title
            # Clear deeper levels
            for j in range(level, 6):
                if j > level - 1:
                    current_hierarchy[j] = None

            # Store hierarchy only for this heading position
            active_hierarchy = [h for h in current_hierarchy if h is not None]
            hierarchy_map[position] = active_hierarchy.copy()

        return hierarchy_map

    def _find_chunk_headings(
        self,
        chunk_text: str,
        original_content: str,
        hierarchy_map: Dict[int, List[str]],
    ) -> List[str]:
        """
        Find the relevant headings for a given chunk based on its position in the original content.
        Uses binary search for efficient lookup of the closest heading position.
        """
        # Find the position of this chunk in the original content
        chunk_start = original_content.find(chunk_text)
        if chunk_start == -1:
            # If exact match not found, try to find the best match
            # This can happen due to text processing differences
            return []

        # Get sorted heading positions for binary search
        heading_positions = sorted(hierarchy_map.keys())

        if not heading_positions:
            return []

        # Find the closest heading position that is <= chunk_start
        # Using binary search for efficiency
        left, right = 0, len(heading_positions) - 1
        best_position = -1

        while left <= right:
            mid = (left + right) // 2
            if heading_positions[mid] <= chunk_start:
                best_position = heading_positions[mid]
                left = mid + 1
            else:
                right = mid - 1

        # Return the hierarchy for the best position found
        if best_position != -1:
            return hierarchy_map[best_position]
        else:
            return []

    async def parse(self, knowledge, content: Text):
        split_config = knowledge.split_config

        # Extract headings and build hierarchy
        headings = self._extract_headings(content.content)
        hierarchy_map = self._build_heading_hierarchy(headings)

        separators = split_config.separators or [
            "\n#{1,6} ",
            "```\n",
            "\n\\*\\*\\*+\n",
            "\n---+\n",
            "\n___+\n",
            "\n\n",
            "\n",
            " ",
            "",
        ]
        if "" not in separators:
            separators.append("")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=split_config.chunk_size,
            chunk_overlap=split_config.chunk_overlap,
            separators=separators,
            keep_separator=False,
        )

        result = []

        # extract all image urls and alt text
        image_pattern = r"!\[(.*?)\]\((.*?)\)"
        all_image_matches = re.findall(image_pattern, content.content)

        # create image objects
        for img_idx, (alt_text, img_url) in enumerate(all_image_matches):
            if img_url.strip():  # ensure url is not empty
                img_metadata = content.metadata.copy()
                img_metadata["_img_idx"] = img_idx
                img_metadata["_img_url"] = img_url.strip()
                img_metadata["_alt_text"] = alt_text.strip() if alt_text.strip() else ""
                image_obj = Image(url=img_url.strip(), metadata=img_metadata)
                result.append(image_obj)

        # split text
        split_texts = splitter.split_text(content.content)

        for idx, text in enumerate(split_texts):
            metadata = content.metadata.copy()
            metadata["_idx"] = idx

            # Find relevant headings for this chunk
            chunk_headings = self._find_chunk_headings(
                text, content.content, hierarchy_map
            )

            # Build context with knowledge name and headings
            context_parts = []
            if knowledge.knowledge_name:
                context_parts.append(knowledge.knowledge_name)
                metadata["_knowledge_name"] = knowledge.knowledge_name

            if chunk_headings:
                metadata["_headings"] = chunk_headings
                metadata["_heading_path"] = " > ".join(chunk_headings)
                context_parts.extend(chunk_headings)

            # Add context information to content
            if context_parts:
                full_context = " > ".join(context_parts)

                # Prepend context to the chunk content for better retrieval
                # This creates a more context-rich chunk that includes knowledge name and hierarchical path
                enhanced_content = f"[Context: {full_context}]\n\n{text}"
                result.append(Text(content=enhanced_content, metadata=metadata))
            else:
                result.append(Text(content=text, metadata=metadata))

        return result


async def test_heading_context():
    """测试标题上下文功能"""

    # 测试用的 Markdown 内容
    markdown_content = """# 🧩 发布特性一览
## ❤️ 喜欢并收藏 优秀UI
允许用户点赞并收藏精彩的UI生成结果，构建个人灵感库

**「朕已阅，甚欢，赐尔收藏」**


![喜欢并收藏功能演示](https://example.com/image1.gif)

## ✍️ 需求自动扩写
用户首次对话的需求描述过于简单时，自动扩写用户需求，智能补全设计意图，生成更完善的UI界面。

**「读心术MAX」—— 终于有AI懂我的意思了，自动匹配一个产品经理**

![需求自动扩写功能演示](https://example.com/image2.gif)

## 🎨 设计规范自动从设计稿中抽取
用户上传了设计稿但没有填写设计规范时，智能分析上传图片，自动提取和应用设计规范，保持品牌一致性

**「设计师掉线检测器」—— 一键破解设计师暗藏的玄机，连Sketch都羡慕的抽取技能。**

![设计规范抽取功能演示](https://example.com/image3.gif)

### 子功能详解
这里是子功能的详细说明内容。

## 自动取名
在生成UI过程中，智能为UI和对话场景赋予恰当且有意义（卧虎AI 很调皮）的名称。

**「告别命名危机」—— 比程序员起变量名靠谱多了**

![](https://example.com/image4.png)
"""

    # 创建一个简单的 mock Knowledge 对象
    class MockKnowledge:
        def __init__(self):
            self.knowledge_name = "UI设计特性文档"  # 添加知识名称
            self.split_config = YuqueSplitConfig(
                chunk_size=500,  # 较小的chunk_size以便测试
                chunk_overlap=100,
                separators=[
                    "\n#{1,6} ",
                    "```\n",
                    "\n\\*\\*\\*+\n",
                    "\n---+\n",
                    "\n___+\n",
                    "\n\n",
                    "\n",
                    " ",
                    "",
                ],
                is_separator_regex=True,
            )

    knowledge = MockKnowledge()

    # 创建测试用的 Text 对象
    text_content = Text(
        content=markdown_content, metadata={"source": "test", "title": "发布特性一览"}
    )

    # 创建解析器实例
    parser = YuqueParser()

    # 执行解析
    result = await parser.parse(knowledge, text_content)

    # 分析结果
    text_objects = [item for item in result if hasattr(item, "content")]
    image_objects = [item for item in result if hasattr(item, "url")]

    print(f"解析结果统计:")
    print(f"- 文本块数量: {len(text_objects)}")
    print(f"- 图片数量: {len(image_objects)}")
    print()

    print("提取的图片:")
    for i, img in enumerate(image_objects):
        print(f"  {i+1}. URL: {img.url}")
        print(f"     描述: {img.metadata.get('_alt_text', '无描述')}")
        print(f"     图片URL: {img.metadata.get('_img_url', 'N/A')}")
        print(f"     图片索引: {img.metadata.get('_img_idx', 'N/A')}")
        print()

    for i, text_obj in enumerate(text_objects[:5]):  # 显示前5个文本块
        print(f"  {i+1}. 长度: {len(text_obj.content)} 字符")
        print(f"     内容预览: {text_obj.content[:150]}...")

        # 显示新增的上下文信息
        metadata = text_obj.metadata
        if "_knowledge_name" in metadata:
            print(f"     知识名称: {metadata['_knowledge_name']}")
        if "_headings" in metadata:
            print(f"     标题层级: {metadata['_headings']}")
        if "_heading_path" in metadata:
            print(f"     标题路径: {metadata['_heading_path']}")

        # 检查内容是否包含上下文标记
        if text_obj.content.startswith("[Context:"):
            context_end = text_obj.content.find("]\n\n")
            if context_end != -1:
                context_info = text_obj.content[
                    9:context_end
                ]  # 提取 [Context: 和 ] 之间的内容
                print(f"     内容中的上下文: {context_info}")

        print(
            f"     其他元数据: {dict((k, v) for k, v in metadata.items() if not k.startswith('_'))}"
        )
        print()

    # 验证标题提取功能
    print("标题提取验证:")
    headings = parser._extract_headings(markdown_content)
    print(f"提取到的标题数量: {len(headings)}")
    for level, title, pos in headings:
        print(f"  级别{level}: {title} (位置: {pos})")
    print()

    # 验证层级构建功能
    print("标题层级构建验证:")
    hierarchy_map = parser._build_heading_hierarchy(headings)
    print(f"层级映射条目数: {len(hierarchy_map)}")
    # 显示几个关键位置的层级信息
    sample_positions = sorted(hierarchy_map.keys())[:5]
    for pos in sample_positions:
        print(f"  位置 {pos}: {hierarchy_map[pos]}")
    print()


if __name__ == "__main__":
    asyncio.run(test_heading_context())
