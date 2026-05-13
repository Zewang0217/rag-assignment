"""
Step 4: 检索
本步骤需要你补全 retrieve() 函数的核心检索逻辑。

任务说明:
    将用户问题转换为向量，在向量数据库中搜索最相似的文本块。
"""

from config import TOP_K
import numpy as np


def retrieve(query: str, model, index, chunks, top_k: int = TOP_K):
    """
    根据用户问题检索最相关的文本块

    参数:
        query: 用户问题（字符串）
        model: SentenceTransformer 模型实例
        index: FAISS 索引实例
        chunks: 文本块列表 List[str]
        top_k: 召回的最相关文本块数量

    返回:
        results: List[dict]，每个 dict 包含:
            - chunk: 文本内容
            - score: 相似度分数（距离越小越相似）
            - rank: 排名（1, 2, 3...）
    """
    # 1. 将用户问题转换为向量
    # ========== TODO: 查询向量化 ==========
    # 提示：使用 model.encode([query])
    # 代码：query_vector = ???

    pass  # 删除这行

    # 2. 在向量数据库中搜索最相似的 top_k 个块
    # ========== TODO: 向量相似度搜索 ==========
    # 提示：使用 index.search(query_vector, top_k)
    # 返回的 distances 是距离矩阵，indices 是对应的索引
    # 代码：distances, indices = ???

    pass  # 删除这行

    # 3. 整理结果，组装成列表返回
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):  # 确保索引不越界
            results.append({
                "chunk": chunks[idx],
                "score": float(distances[0][i]),
                "rank": i + 1
            })

    return results


def format_results(results, max_preview_len: int = 100):
    """
    格式化检索结果，便于展示（已提供）
    """
    if not results:
        return "未找到相关结果"

    output = []
    for r in results:
        preview = r["chunk"][:max_preview_len].replace("\n", " ")
        output.append(
            f"[Rank {r['rank']}] 分数: {r['score']:.4f}\n"
            f"  内容: {preview}..."
        )
    return "\n".join(output)


if __name__ == "__main__":
    # 测试检索功能
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    from config import KNOWLEDGE_BASE_PATH

    print("=" * 40)
    print("Step 4 测试")
    print("=" * 40)

    # 加载组件
    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, _ = create_vector_store(chunks)

    # 测试查询
    test_queries = [
        "社团的宗旨是什么？",
        "会员费是多少钱？",
    ]

    for q in test_queries:
        print(f"\n查询: {q}")
        results = retrieve(q, model, index, chunks)
        print(format_results(results))

    print("\n测试通过！")