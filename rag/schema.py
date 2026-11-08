"""数据结构定义：整个项目里流转的"数据长什么样"。

单独放一个文件的好处：
1. 其他模块 import 时不会产生循环依赖；
2. 数据结构集中在一处，改字段时只需要改这里。
"""

from dataclasses import dataclass


@dataclass
class Document:
    """一篇知识库文档（阶段 1 里 = 一个 txt 文件）。

    注意：阶段 1 还没有做切分，所以"文档"就是文件本身。
    阶段 2 引入 chunking 之后，这里会多出 chunk 相关的字段。
    """

    doc_id: str      # 唯一标识，这里直接用文件名（不含扩展名）
    title: str       # 展示给人看的标题，用于答案里的来源引用
    text: str        # 文档正文


@dataclass
class SearchResult:
    """一次检索命中的结果：命中了哪篇文档 + 相似度得分。"""

    document: Document
    score: float
