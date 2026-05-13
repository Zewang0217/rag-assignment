"""
Step 2: 文本切分
本步骤需要你补全 chunk_text() 函数的核心逻辑。

任务说明:
    实现一个滑动窗口切分器，将长文本切分成固定大小的文本块。
    相邻块之间要有重叠(overlap)，以保持上下文连贯性。
"""

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """
    将长文本切分为小块，使用滑动窗口方式

    参数:
        text: 原始文本
        chunk_size: 每个文本块的字符数
        chunk_overlap: 相邻块之间的重叠字符数

    返回:
        chunks: List[str]，文本块列表

    示例:
        text = "ABCDEFGHIJ" (10个字符)
        chunk_size = 4, chunk_overlap = 2
        期望结果: ["ABCD", "CDEF", "EFGH", "GHIJ"]
    """
    chunks = []
    start = 0
    text_len = len(text)

    # ========== TODO: 补全以下逻辑 ==========
    # 提示：
    # 1. 使用 while 循环，从文本开头一直切到末尾
    # 2. 每次切出一个 [start, start + chunk_size) 的片段
    # 3. 跳过过短的片段（少于 20 字符的丢弃）
    # 4. 每切完一块，start 向后移动 (chunk_size - chunk_overlap)
    # 5. 直到 start >= text_len 结束

    # 你的代码：

    pass  # 删除这行，填入你的代码


def get_chunk_stats(chunks):
    """辅助函数：统计文本块信息（已提供）"""
    if not chunks:
        return {"count": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

    lengths = [len(c) for c in chunks]
    return {
        "count": len(chunks),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths)
    }


if __name__ == "__main__":
    # 测试代码
    from step1_loader import load_document
    from config import KNOWLEDGE_BASE_PATH

    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)

    stats = get_chunk_stats(chunks)
    print(f"切分结果统计: {stats}")
    print(f"总块数: {stats['count']}")

    # 验证：每个块的大小应该在合理范围内
    if stats['min_length'] < 20:
        print("警告：存在过短的文本块")
    else:
        print("文本块长度正常")
