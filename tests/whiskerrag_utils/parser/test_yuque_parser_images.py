#!/usr/bin/env python3
"""
测试 YuqueParser 的图片提取功能
"""
import asyncio
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from whiskerrag_types.model.multi_modal import Text
from whiskerrag_types.model.splitter import YuqueSplitConfig
from whiskerrag_utils.parser.yuque_parser import YuqueParser


async def test_image_extraction():
    """测试图片提取功能"""

    # 测试用的 Markdown 内容
    markdown_content = """# 🧩 发布特性一览
## <font style="color:rgb(13, 18, 57);">❤️</font><font style="color:rgb(13, 18, 57);"> 喜欢并收藏 优秀UI</font>
允许用户点赞并收藏精彩的UI生成结果，构建个人灵感库

**「朕已阅，甚欢，赐尔收藏」**

![喜欢并收藏功能演示](https://intranetproxy.alipay.com/skylark/lark/0/2025/gif/117356539/1751270678905-c7b3686b-6429-443d-b91d-11802a20bab3.gif)

## <font style="color:rgb(13, 18, 57);">✍️</font><font style="color:rgb(13, 18, 57);"> 需求自动扩写</font>
用户首次对话的需求描述过于简单时，自动扩写用户需求，智能补全设计意图，生成更完善的UI界面。

**「读心术MAX」—— 终于有AI懂我的意思了，自动匹配一个产品经理**

![需求自动扩写功能演示](https://intranetproxy.alipay.com/skylark/lark/0/2025/gif/117356539/1751270936881-92d97d6f-f746-4258-aca0-41fa47c7b0ca.gif)

## <font style="color:rgb(13, 18, 57);">🎨</font><font style="color:rgb(13, 18, 57);"> 设计规范自动从设计稿中抽取</font>
<font style="color:rgb(13, 18, 57);">用户上传了设计稿但没有填写设计规范时，智能分析上传图片，自动提取和应用设计规范，保持品牌一致性</font>

**「设计师掉线检测器」—— 一键破解设计师暗藏的玄机，连Sketch都羡慕的抽取技能。 **

![设计规范抽取功能演示](https://intranetproxy.alipay.com/skylark/lark/0/2025/gif/117356539/1751271336383-17528d53-f8d3-460a-a37d-a748ce47e165.gif)

## <font style="color:rgb(13, 18, 57);">自动取名</font>
<font style="color:rgb(13, 18, 57);">在生成UI过程中，智能为UI和对话场景赋予恰当且有意义（卧虎AI 很调皮）的名称。</font>

**「告别命名危机」—— 比程序员起变量名靠谱多了**

![](https://intranetproxy.alipay.com/skylark/lark/0/2025/png/117356539/1751271641847-76fac152-50d9-454a-a606-5bfc3a3a3cf5.png)

## <font style="color:rgb(13, 18, 57);">bug修复</font>
619前瞻版本的一些bug修复和交互体验优化

****

## <font style="color:rgb(13, 18, 57);">✨</font><font style="color:rgb(13, 18, 57);"> 前瞻：GenUI Agent 与 素材生成</font>
集成式GenUI接口服务，自动规划UI生成步骤，智能调用各类设计工具，同时集成素材生成与检索功能，让UI设计更加丰富多彩

**「UI全栈自动驾驶」—— 转行契机已到，这AI不仅会"做饭"还会自己"买菜」。**
"""

    # 创建一个简单的 mock Knowledge 对象
    class MockKnowledge:
        def __init__(self):
            self.split_config = YuqueSplitConfig(
                chunk_size=1000,
                chunk_overlap=200,
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
        print(f"     元数据: {img.metadata}")
        print()

    print("文本块预览:")
    for i, text_obj in enumerate(text_objects[:3]):  # 只显示前3个
        print(f"  {i+1}. 长度: {len(text_obj.content)} 字符")
        print(f"     内容预览: {text_obj.content[:100]}...")
        print()


if __name__ == "__main__":
    asyncio.run(test_image_extraction())
