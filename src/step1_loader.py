"""
Step 1: 加载文档
本步骤已完整提供，无需修改。
"""

import os

def load_document(path: str) -> str:
    """
    从指定路径加载文本文件

    参数:
        path: 文件路径

    返回:
        文件内容（字符串）
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    # 测试加载
    from config import KNOWLEDGE_BASE_PATH
    text = load_document(KNOWLEDGE_BASE_PATH)
    print(f"加载成功，字数: {len(text)}")
    print("前100字预览:", text[:100])
