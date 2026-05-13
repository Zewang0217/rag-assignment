"""
自动评分测试 - 参考实现版
使用 rag_solution.py 中已完成的实现来验证评分逻辑
运行方式: pytest tests/test_reference.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 导入参考实现（完整版本）
from rag_solution import load_document, chunk_text, create_vector_store, retrieve, generate_answer
from config import KNOWLEDGE_BASE_PATH, CHUNK_SIZE, CHUNK_OVERLAP


class TestStep1Loader:
    """Step 1: 文档加载 (15分)"""

    def test_file_exists(self):
        assert os.path.exists(KNOWLEDGE_BASE_PATH)

    def test_load_returns_string(self):
        text = load_document(KNOWLEDGE_BASE_PATH)
        assert isinstance(text, str)

    def test_load_not_empty(self):
        text = load_document(KNOWLEDGE_BASE_PATH)
        assert len(text) > 0

    def test_load_encoding(self):
        text = load_document(KNOWLEDGE_BASE_PATH)
        assert any(kw in text for kw in ["博远", "社团", "会员", "社长"])


class TestStep2Chunker:
    """Step 2: 文本切分 (25分)"""

    def test_chunk_returns_list(self):
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        assert isinstance(chunks, list)

    def test_chunk_not_empty(self):
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        assert len(chunks) > 0

    def test_chunk_count_reasonable(self):
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        max_expected = len(text) // 20 + 2
        assert 2 <= len(chunks) <= max_expected, f"块数 {len(chunks)} 不合理"

    def test_chunk_size_reasonable(self):
        from rag_solution import get_chunk_stats
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        stats = get_chunk_stats(chunks)
        assert stats["avg_length"] >= CHUNK_SIZE * 0.5
        assert stats["avg_length"] <= CHUNK_SIZE * 1.5

    def test_chunk_has_overlap(self):
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        if len(chunks) >= 2:
            has_overlap = False
            for i in range(len(chunks) - 1):
                overlap_chars = set(chunks[i]) & set(chunks[i + 1])
                if len(overlap_chars) >= CHUNK_OVERLAP:
                    has_overlap = True
                    break
            assert has_overlap


class TestStep3Embedder:
    """Step 3: 向量化存储 (30分)"""

    def test_vectorstore_returns_correct_types(self):
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)[:5]
        result = create_vector_store(chunks)
        assert len(result) == 3
        model, index, returned_chunks = result
        assert model is not None
        assert index is not None

    def test_vector_dimension(self):
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)[:3]
        model, index, _ = create_vector_store(chunks)
        expected_dim = model.get_sentence_embedding_dimension()
        import numpy as np
        dummy = np.random.randn(1, expected_dim).astype("float32")
        distances, indices = index.search(dummy, 1)
        assert distances.shape == (1, 1)

    def test_index_has_vectors(self):
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)[:5]
        _, index, _ = create_vector_store(chunks)
        assert index.ntotal == len(chunks)


class TestStep4Retriever:
    """Step 4: 检索 (20分)"""

    @pytest.fixture
    def setup(self):
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        model, index, _ = create_vector_store(chunks)
        return {"model": model, "index": index, "chunks": chunks}

    def test_retrieve_returns_list(self, setup):
        results = retrieve("社团宗旨", setup["model"], setup["index"], setup["chunks"])
        assert isinstance(results, list)

    def test_retrieve_not_empty(self, setup):
        results = retrieve("社团宗旨", setup["model"], setup["index"], setup["chunks"])
        assert len(results) > 0

    def test_retrieve_has_required_fields(self, setup):
        results = retrieve("会员费", setup["model"], setup["index"], setup["chunks"])
        r = results[0]
        assert all(k in r for k in ["chunk", "score", "rank"])

    def test_retrieve_top_k_respected(self, setup):
        results = retrieve("活动", setup["model"], setup["index"], setup["chunks"], top_k=2)
        assert len(results) <= 2

    def test_retrieve_relevance(self, setup):
        results = retrieve("会员费是多少钱", setup["model"], setup["index"], setup["chunks"])
        top = results[0]["chunk"]
        # 检查检索结果是否与"会员费"相关（允许相关语义词）
        assert any(kw in top for kw in ["费", "50", "社费", "元", "每", "学期"]), \
            f"检索结果可能不相关: {top[:100]}"


class TestStep5Generator:
    """Step 5: 生成回答 (10分)"""

    @pytest.fixture
    def setup(self):
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        model, index, _ = create_vector_store(chunks)
        return retrieve("会员费", model, index, chunks)

    def test_generate_returns_string(self, setup):
        answer = generate_answer("会员费多少", setup)
        assert isinstance(answer, str)

    def test_generate_handles_empty(self):
        answer = generate_answer("完全不相关的问题", [])
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_generate_uses_context(self, setup):
        answer = generate_answer("社团宗旨是什么", setup)
        assert len(answer) > 10


class TestIntegration:
    """综合测试 (加分项)"""

    def test_end_to_end(self):
        from rag_solution import rag_pipeline
        answer, results = rag_pipeline("社团的宗旨是什么？")
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert len(results) > 0

    def test_multiple_queries(self):
        from rag_solution import rag_pipeline
        for q in ["会员费是多少？", "社长如何联系？", "每周活动什么时候？"]:
            answer, results = rag_pipeline(q)
            assert len(answer) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])