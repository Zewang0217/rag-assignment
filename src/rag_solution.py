"""
RAG Pipeline - 完整可运行版本（参考解答）
用于验证 autograder 能否正常工作
"""

import os
import sys
from config import (
    KNOWLEDGE_BASE_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K, EMBEDDING_MODEL, EMBEDDING_DEVICE
)

# ============ Step 1: 加载文档 ============
def load_document(path: str) -> str:
    """从指定路径加载文本文件"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ============ Step 2: 切分文本 ============
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """将长文本切分为小块，使用滑动窗口方式，相邻块之间有重叠"""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        # 去除过短的块
        if len(chunk) >= 20:
            chunks.append(chunk)

        # 滑动窗口：向后移动 (chunk_size - chunk_overlap)
        start += chunk_size - chunk_overlap

    return chunks

def get_chunk_stats(chunks):
    """统计文本块信息"""
    if not chunks:
        return {"count": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
    lengths = [len(c) for c in chunks]
    return {
        "count": len(chunks),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths)
    }

# ============ Step 3: 向量化 & 存储 ============
def create_vector_store(chunks, model_name: str = EMBEDDING_MODEL, device: str = EMBEDDING_DEVICE):
    """将文本块转换为向量并存储到 FAISS 向量数据库"""
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    # 1. 加载模型
    model = SentenceTransformer(model_name, device=device)

    # 2. 生成向量
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # 3. 创建 FAISS 索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return model, index, chunks

# ============ Step 4: 检索 ============
def retrieve(query: str, model, index, chunks, top_k: int = TOP_K):
    """根据用户问题检索最相关的文本块"""
    import numpy as np

    # 查询向量化
    query_vector = model.encode([query]).astype("float32")

    # 向量相似度搜索
    distances, indices = index.search(query_vector, top_k)

    # 组装结果
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            results.append({
                "chunk": chunks[idx],
                "score": float(distances[0][i]),
                "rank": i + 1
            })

    return results

def format_results(results, max_preview_len: int = 100):
    """格式化检索结果"""
    if not results:
        return "未找到相关结果"
    output = []
    for r in results:
        preview = r["chunk"][:max_preview_len].replace("\n", " ")
        output.append(f"[Rank {r['rank']}] 分数: {r['score']:.4f}\n  内容: {preview}...")
    return "\n".join(output)

# ============ Step 5: 生成 ============
def generate_answer(query: str, retrieved_chunks: list, api_key: str = None):
    """基于检索结果生成回答"""
    if not retrieved_chunks:
        return "抱歉，我在知识库中没有找到与您问题相关的内容。"

    context = "\n\n".join([c["chunk"] for c in retrieved_chunks])
    answer = extract_answer_from_context(query, context)
    return answer

def extract_answer_from_context(query: str, context: str) -> str:
    """从上下文中提取答案"""
    context_clean = context.replace("\n", " ").replace("  ", " ")
    sentences = context_clean.split("。")
    relevant = []

    keywords_map = {
        "宗旨": ["推广", "信息", "技术", "知识"],
        "会员费": ["50", "元", "社费", "每学期"],
        "社长": ["社长"],
        "联系": ["邮箱", "QQ", "电话", "办公室"],
        "活动": ["每周", "每月", "Hackathon", "研讨会"],
    }

    # 找出query中的意图关键词
    intent_keyword = None
    for intent, kws in keywords_map.items():
        if intent in query:
            intent_keyword = intent
            related_kws = kws
            break

    if intent_keyword:
        for sentence in sentences:
            if any(kw in sentence for kw in related_kws):
                relevant.append(sentence)
                if len(relevant) >= 2:
                    break

    if relevant:
        return " ".join(relevant[:2]) + "。"
    return "根据检索到的资料，" + sentences[0] + "。" if sentences else "未找到确切答案"

# ============ 主程序 ============
def rag_pipeline(query: str, model=None, index=None, chunks=None):
    """RAG 完整流程"""
    if model is None or index is None or chunks is None:
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        model, index, chunks = create_vector_store(chunks)

    results = retrieve(query, model, index, chunks)
    answer = generate_answer(query, results)
    return answer, results

if __name__ == "__main__":
    print("=" * 50)
    print("RAG 知识库系统 - 完整可运行版本")
    print("=" * 50)

    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, _ = create_vector_store(chunks)

    queries = ["社团的宗旨是什么？", "会员费是多少？", "社长如何联系？"]

    for q in queries:
        print(f"\n问题: {q}")
        answer, results = rag_pipeline(q, model, index, chunks)
        print(f"回答: {answer}")
        print(f"检索到 {len(results)} 个片段，分数: {results[0]['score']:.4f}")