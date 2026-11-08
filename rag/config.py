"""全局配置：环境变量加载 + 路径常量。

只用标准库手写 .env 解析，帮你搞明白 python-dotenv 到底做了什么：
读文件 -> 逐行 -> 跳过注释和空行 -> 按第一个 "=" 切成键值对 -> 写进 os.environ。
"""

import os
from pathlib import Path

# 项目根目录 = 本文件所在目录的上一级（rag/config.py -> ragxuexi/）
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


def load_env_file(path: Path | None = None) -> None:
    """把 .env 里的键值对写进 os.environ（已存在的环境变量不覆盖）。"""
    env_path = path or (ROOT_DIR / ".env")

    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        # 跳过空行和注释行
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        # 只按第一个 "=" 切分，这样 value 里可以包含 "="
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        # setdefault：系统环境变量优先于 .env，方便在 Docker/CI 里注入
        os.environ.setdefault(key, value)


# 模块被 import 时自动加载一次
load_env_file()

# 检索默认召回几篇文档。数字越大，模型看到的资料越多，但 Prompt 也越长、越容易被无关内容带偏
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "2"))

# 最低相似度阈值：得分不高于它的文档不会进入 Prompt。
# 为什么默认是 0（即"至少有一个 token 重合"）？
#   余弦相似度的绝对值受文档长度、问题长度影响，不存在跨项目通用的数字。
#   这里只做最低限度的事：把"字面完全不沾边"的情况拦下来。
# 真实项目必须用评测集来标定这个阈值 —— 阶段 7 会正式做。
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0"))
