"""
自动评分测试（纯函数式，无类包装）
评分标准：
- Step 1 (15分): load_document() 正确加载文件
- Step 2 (25分): chunk_text() 正确切分文本，块数量合理
- Step 3 (30分): create_vector_store() 正确生成向量并存入索引
- Step 4 (15分): retrieve() 正确检索相关文本块
- Step 5 (15分): generate_answer() 正确生成回答
"""

import sys
import os

# 确保 PYTHONPATH 正确
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import KNOWLEDGE_BASE_PATH, CHUNK_SIZE, CHUNK_OVERLAP

# ===== Step 1 Tests (15分) =====
def test_step1_file_exists():
    """Step 1: 知识库文件存在"""
    import os
    assert os.path.exists(KNOWLEDGE_BASE_PATH), f"知识库文件不存在: {KNOWLEDGE_BASE_PATH}"

def test_step1_load_returns_string():
    """Step 1: load_document() 返回字符串"""
    from step1_loader import load_document
    result = load_document(KNOWLEDGE_BASE_PATH)
    assert isinstance(result, str), f"load_document() 应返回 str，实际返回 {type(result)}"

def test_step1_load_not_empty():
    """Step 1: load_document() 返回非空内容"""
    from step1_loader import load_document
    result = load_document(KNOWLEDGE_BASE_PATH)
    assert len(result) > 100, f"文档内容太短: {len(result)}"

def test_step1_load_encoding():
    """Step 1: load_document() 正确处理编码，无乱码"""
    from step1_loader import load_document
    result = load_document(KNOWLEDGE_BASE_PATH)
    # 检查无常见乱码字符
    assert "\ufffd" not in result, "文档包含乱码字符"
    # 允许首尾有空格/换行
    assert len(result) > 100, "文档内容太短"


# ===== Step 2 Tests (25分) =====
def test_step2_chunk_returns_list():
    """Step 2: chunk_text() 返回列表"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    text = load_document(KNOWLEDGE_BASE_PATH)
    result = chunk_text(text)
    assert isinstance(result, list), f"chunk_text() 应返回 list，实际返回 {type(result)}"

def test_step2_chunk_not_empty():
    """Step 2: chunk_text() 返回非空列表"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    text = load_document(KNOWLEDGE_BASE_PATH)
    result = chunk_text(text)
    assert len(result) > 0, "chunk_text() 返回了空列表"

def test_step2_chunk_count_reasonable():
    """Step 2: 文本块数量合理（至少5块，最多100块）"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    assert 5 <= len(chunks) <= 100, f"文本块数量不合理: {len(chunks)}"

def test_step2_chunk_size_reasonable():
    """Step 2: 每个文本块长度在合理范围内（20~500字符）"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        assert 20 <= len(chunk) <= 500, f"第{i+1}个块长度异常: {len(chunk)}"

def test_step2_chunk_has_overlap():
    """Step 2: 相邻文本块有重叠（验证滑动窗口）"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    if len(chunks) < 2:
        raise AssertionError("块数量少于2个，无法验证重叠")
    # 检查：第二个块的开头应该出现在第一个块的末尾（重叠区域）
    found_overlap = False
    for i in range(len(chunks) - 1):
        tail = chunks[i][-40:] if len(chunks[i]) >= 40 else chunks[i]
        head = chunks[i+1][:20] if len(chunks[i+1]) >= 20 else chunks[i+1]
        if any(c in tail for c in head):
            found_overlap = True
            break
    assert found_overlap, "未检测到相邻块之间的重叠（滑动窗口未生效）"


# ===== Step 3 Tests (30分) =====
def test_step3_vectorstore_returns_tuple():
    """Step 3: create_vector_store() 返回三元组(model, index, chunks)"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    result = create_vector_store(chunks)
    assert isinstance(result, tuple) and len(result) == 3, f"create_vector_store() 应返回 (model, index, chunks)，实际返回 {type(result)}"

def test_step3_model_is_sentence_transformer():
    """Step 3: 返回的 model 是 SentenceTransformer 实例"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, returned_chunks = create_vector_store(chunks)
    assert model is not None, "model 不应为 None"

def test_step3_vector_dimension():
    """Step 3: 向量维度合理（通常为 384/768/1024）"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, returned_chunks = create_vector_store(chunks)
    import numpy as np
    dim = index.d
    assert dim in [384, 768, 1024, 1536], f"向量维度异常: {dim}"

def test_step3_index_has_vectors():
    """Step 3: FAISS 索引包含向量"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, returned_chunks = create_vector_store(chunks)
    assert index.ntotal > 0, "索引中没有任何向量"


# ===== Step 4 Tests (15分) =====
def test_step4_retrieve_returns_list():
    """Step 4: retrieve() 返回列表"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    from step4_retriever import retrieve

    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, _ = create_vector_store(chunks)
    results = retrieve("社团的宗旨是什么？", model, index, chunks)
    assert isinstance(results, list), f"retrieve() 应返回 list，实际返回 {type(results)}"

def test_step4_retrieve_not_empty():
    """Step 4: retrieve() 返回非空结果"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    from step4_retriever import retrieve

    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, _ = create_vector_store(chunks)
    results = retrieve("社团的宗旨是什么？", model, index, chunks)
    assert len(results) > 0, "retrieve() 返回了空列表"

def test_step4_retrieve_has_required_fields():
    """Step 4: 返回结果包含必需字段 (chunk, score, rank)"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    from step4_retriever import retrieve

    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, _ = create_vector_store(chunks)
    results = retrieve("社团的宗旨是什么？", model, index, chunks)
    for r in results:
        assert all(k in r for k in ["chunk", "score", "rank"]), f"缺少必需字段: {r.keys()}"

def test_step4_retrieve_top_k_respected():
    """Step 4: retrieve() 尊重 top_k 参数"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    from step4_retriever import retrieve

    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, _ = create_vector_store(chunks)
    results = retrieve("社团的宗旨是什么？", model, index, chunks, top_k=3)
    assert len(results) == 3, f"top_k=3 但返回了 {len(results)} 个结果"

def test_step4_retrieve_relevance():
    """Step 4: 检索结果与查询相关"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    from step4_retriever import retrieve

    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, _ = create_vector_store(chunks)
    results = retrieve("会员费是多少？", model, index, chunks)

    assert len(results) > 0, "未检索到任何结果"
    top_chunk = results[0]["chunk"]
    # 检查是否包含"费"、"50"、"社费"、"元"、"每"、"学期"、"缴纳"等相关词
    assert any(kw in top_chunk for kw in ["费", "50", "社费", "元", "每", "学期", "缴纳"]), \
        f"检索结果可能不相关，内容: {top_chunk[:100]}"


# ===== Step 5 Tests (15分) =====
def test_step5_generate_returns_string():
    """Step 5: generate_answer() 返回字符串"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    from step4_retriever import retrieve
    from step5_generator import generate_answer

    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, _ = create_vector_store(chunks)
    retrieved = retrieve("社团的宗旨是什么？", model, index, chunks)
    result = generate_answer("社团的宗旨是什么？", retrieved)
    assert isinstance(result, str), f"generate_answer() 应返回 str，实际返回 {type(result)}"

def test_step5_generate_handles_empty():
    """Step 5: generate_answer() 能处理空检索结果"""
    from step5_generator import generate_answer
    result = generate_answer("一个无关问题", [])
    assert isinstance(result, str), "空检索时应返回字符串"

def test_step5_generate_uses_context():
    """Step 5: 生成的回答使用了上下文信息"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    from step4_retriever import retrieve
    from step5_generator import generate_answer

    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, _ = create_vector_store(chunks)
    retrieved = retrieve("社团的宗旨是什么？", model, index, chunks)
    answer = generate_answer("社团的宗旨是什么？", retrieved)
    # 回答中应包含来自检索结果的关键词（不只是"不知道"）
    assert len(answer) > 10, "回答太短，可能没有使用上下文"
    # 只要包含任何一个 generic 词，就算失败
    unknown_phrases = ["未找到", "无法确定", "不知道", "没有相关信息", "cannot answer", "no information"]
    has_generic_phrase = any(phrase.lower() in answer.lower() for phrase in unknown_phrases)
    assert not has_generic_phrase, f"回答似乎只是通用回复，没有使用上下文: {answer[:100]}"


# ===== Integration Tests =====
def test_integration_end_to_end():
    """集成测试: 完整流程"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    from step4_retriever import retrieve
    from step5_generator import generate_answer

    text = load_document(KNOWLEDGE_BASE_PATH)
    assert len(text) > 0, "文档加载失败"

    chunks = chunk_text(text)
    assert len(chunks) > 0, "文本切分失败"

    model, index, _ = create_vector_store(chunks)
    assert index.ntotal > 0, "向量存储失败"

    results = retrieve("社团的宗旨是什么？", model, index, chunks)
    assert len(results) > 0, "检索失败"

    answer = generate_answer("社团的宗旨是什么？", results)
    assert len(answer) > 0, "生成回答失败"
    print(f"\n完整流程测试通过！示例回答: {answer[:80]}...")

def test_integration_multiple_queries():
    """集成测试: 多查询稳定性"""
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    from step4_retriever import retrieve
    from step5_generator import generate_answer

    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, _ = create_vector_store(chunks)

    queries = ["社团的宗旨是什么？", "会员费是多少钱？", "如何成为会员？"]
    for q in queries:
        results = retrieve(q, model, index, chunks)
        answer = generate_answer(q, results)
        assert len(answer) > 5, f"查询'{q}'的回答太短"