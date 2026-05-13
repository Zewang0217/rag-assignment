"""
自动评分测试
评分标准：
- Step 1 (15分): load_document() 正确加载文件
- Step 2 (25分): chunk_text() 正确切分文本，块数量合理
- Step 3 (30分): create_vector_store() 向量维度正确
- Step 4 (20分): retrieve() 检索结果与查询相关
- Step 5 (10分): generate_answer() 能基于检索结果生成回答
总分: 100分

运行方式: pytest tests/test_autograder.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import KNOWLEDGE_BASE_PATH, CHUNK_SIZE, CHUNK_OVERLAP


class TestStep1Loader:
    """Step 1: 文档加载 (15分)"""

    def test_file_exists(self):
        """文件存在"""
        assert os.path.exists(KNOWLEDGE_BASE_PATH), f"知识库文件不存在: {KNOWLEDGE_BASE_PATH}"

    def test_load_returns_string(self):
        """加载返回字符串"""
        from step1_loader import load_document
        text = load_document(KNOWLEDGE_BASE_PATH)
        assert isinstance(text, str), "加载应返回字符串"

    def test_load_not_empty(self):
        """文件非空"""
        from step1_loader import load_document
        text = load_document(KNOWLEDGE_BASE_PATH)
        assert len(text) > 0, "文件内容为空"

    def test_load_encoding(self):
        """UTF-8 编码正确，无乱码"""
        from step1_loader import load_document
        text = load_document(KNOWLEDGE_BASE_PATH)
        # 检查是否包含预期的中文内容（社团章程关键词）
        assert any(keyword in text for keyword in ["博远", "社团", "会员", "社长"]), \
            "文件内容可能存在乱码或不匹配"


class TestStep2Chunker:
    """Step 2: 文本切分 (25分)"""

    def test_chunk_returns_list(self):
        """返回列表"""
        from step1_loader import load_document
        from step2_chunker import chunk_text
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        assert isinstance(chunks, list), "应返回列表"

    def test_chunk_not_empty(self):
        """切分结果非空"""
        from step1_loader import load_document
        from step2_chunker import chunk_text
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        assert len(chunks) > 0, "切分结果为空"

    def test_chunk_count_reasonable(self):
        """块数量合理（至少2块，最多不超过原文长/20）"""
        from step1_loader import load_document
        from step2_chunker import chunk_text
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        min_expected = 2
        max_expected = len(text) // 20 + 2
        assert min_expected <= len(chunks) <= max_expected, \
            f"块数量 {len(chunks)} 不在合理范围 [{min_expected}, {max_expected}]"

    def test_chunk_size_reasonable(self):
        """每个块长度在合理范围"""
        from step1_loader import load_document
        from step2_chunker import chunk_text
        from step2_chunker import get_chunk_stats
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        stats = get_chunk_stats(chunks)

        # 块平均长度应该接近 CHUNK_SIZE
        assert stats["avg_length"] >= CHUNK_SIZE * 0.5, \
            f"平均块长度 {stats['avg_length']} 太小"
        assert stats["avg_length"] <= CHUNK_SIZE * 1.5, \
            f"平均块长度 {stats['avg_length']} 太大"

    def test_chunk_has_overlap(self):
        """验证有重叠（相邻块有共同内容）"""
        from step1_loader import load_document
        from step2_chunker import chunk_text
        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)

        if len(chunks) >= 2:
            # 检查是否有共同的字符（验证 overlap 生效）
            has_overlap = False
            for i in range(len(chunks) - 1):
                overlap_chars = set(chunks[i]) & set(chunks[i + 1])
                if len(overlap_chars) >= CHUNK_OVERLAP:
                    has_overlap = True
                    break
            assert has_overlap, "切分结果没有重叠，可能实现有误"


class TestStep3Embedder:
    """Step 3: 向量化存储 (30分)"""

    def test_vectorstore_returns_model_index_chunks(self):
        """返回 (model, index, chunks)"""
        from step1_loader import load_document
        from step2_chunker import chunk_text
        from step3_embedder import create_vector_store
        from config import EMBEDDING_MODEL, EMBEDDING_DEVICE

        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)

        # 只取前5个块测试，加速
        chunks = chunks[:5]

        result = create_vector_store(chunks)
        assert len(result) == 3, "应返回 (model, index, chunks)"
        model, index, returned_chunks = result

        assert model is not None
        assert index is not None
        assert len(returned_chunks) == len(chunks)

    def test_vector_dimension(self):
        """向量维度正确（与模型输出维度一致）"""
        from step1_loader import load_document
        from step2_chunker import chunk_text
        from step3_embedder import create_vector_store
        from sentence_transformers import SentenceTransformer

        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        chunks = chunks[:3]

        model, index, _ = create_vector_store(chunks)

        # 获取模型的标准维度
        expected_dim = model.get_sentence_embedding_dimension()

        # 通过搜索验证维度
        import numpy as np
        dummy_vector = np.random.randn(1, expected_dim).astype("float32")
        distances, indices = index.search(dummy_vector, 1)

        assert distances.shape == (1, 1), "搜索结果维度错误"

    def test_index_has_vectors(self):
        """索引包含向量"""
        from step1_loader import load_document
        from step2_chunker import chunk_text
        from step3_embedder import create_vector_store

        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        chunks = chunks[:5]

        _, index, _ = create_vector_store(chunks)
        assert index.ntotal == len(chunks), f"索引应包含 {len(chunks)} 条向量，实际 {index.ntotal}"


class TestStep4Retriever:
    """Step 4: 检索 (20分)"""

    @pytest.fixture
    def setup_rag(self):
        """共享的 RAG 组件"""
        from step1_loader import load_document
        from step2_chunker import chunk_text
        from step3_embedder import create_vector_store
        from step4_retriever import retrieve

        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        model, index, _ = create_vector_store(chunks)

        return {"model": model, "index": index, "chunks": chunks, "retrieve": retrieve}

    def test_retrieve_returns_list(self, setup_rag):
        """检索返回列表"""
        results = setup_rag["retrieve"]("社团宗旨", setup_rag["model"],
                                       setup_rag["index"], setup_rag["chunks"])
        assert isinstance(results, list), "检索应返回列表"

    def test_retrieve_not_empty(self, setup_rag):
        """检索有结果"""
        results = setup_rag["retrieve"]("社团宗旨", setup_rag["model"],
                                        setup_rag["index"], setup_rag["chunks"])
        assert len(results) > 0, "检索结果为空"

    def test_retrieve_has_required_fields(self, setup_rag):
        """结果包含必要字段"""
        results = setup_rag["retrieve"]("会员费", setup_rag["model"],
                                        setup_rag["index"], setup_rag["chunks"])
        assert len(results) > 0
        r = results[0]
        assert "chunk" in r, "缺少 chunk 字段"
        assert "score" in r, "缺少 score 字段"
        assert "rank" in r, "缺少 rank 字段"

    def test_retrieve_top_k_respected(self, setup_rag):
        """top_k 参数生效"""
        results = setup_rag["retrieve"]("活动", setup_rag["model"],
                                        setup_rag["index"], setup_rag["chunks"], top_k=2)
        assert len(results) <= 2, f"结果数量应为 <= 2，实际 {len(results)}"

    def test_retrieve_relevance(self, setup_rag):
        """检索结果与查询相关（验证语义匹配生效）"""
        # 查询"会员费"应该能召回包含"50"或"社费"的内容
        results = setup_rag["retrieve"]("会员费是多少钱", setup_rag["model"],
                                        setup_rag["index"], setup_rag["chunks"])

        assert len(results) > 0, "未检索到任何结果"

        # 至少第一个结果的 chunk 包含"费"或"50"
        top_chunk = results[0]["chunk"]
        assert any(kw in top_chunk for kw in ["费", "50", "社费", "元"]), \
            f"检索结果不相关，内容: {top_chunk[:100]}"


class TestStep5Generator:
    """Step 5: 生成回答 (10分)"""

    @pytest.fixture
    def setup_rag(self):
        """共享的 RAG 组件"""
        from step1_loader import load_document
        from step2_chunker import chunk_text
        from step3_embedder import create_vector_store
        from step4_retriever import retrieve

        text = load_document(KNOWLEDGE_BASE_PATH)
        chunks = chunk_text(text)
        model, index, _ = create_vector_store(chunks)
        results = retrieve("会员费", model, index, chunks)

        return {"retrieve": retrieve, "model": model, "index": index, "chunks": chunks}

    def test_generate_returns_string(self, setup_rag):
        """生成返回字符串"""
        from step5_generator import generate_answer
        from step4_retriever import retrieve

        results = retrieve("社长", setup_rag["model"], setup_rag["index"], setup_rag["chunks"])
        answer = generate_answer("社长是谁", results)
        assert isinstance(answer, str), "生成应返回字符串"

    def test_generate_handles_empty(self):
        """处理空检索结果"""
        from step5_generator import generate_answer
        answer = generate_answer("完全不相关的问题", [])
        assert isinstance(answer, str)
        assert len(answer) > 0, "空结果时应返回提示语"

    def test_generate_uses_context(self, setup_rag):
        """生成的回答包含检索到的内容"""
        from step5_generator import generate_answer
        from step4_retriever import retrieve

        results = retrieve("社团宗旨", setup_rag["model"], setup_rag["index"], setup_rag["chunks"])
        answer = generate_answer("社团宗旨是什么", results)

        # 回答不应为空
        assert len(answer) > 10, "回答过短，可能未使用上下文"

        # 回答应包含检索结果中的内容特征
        # 正确的回答应该包含社团相关的内容
        assert any(kw in answer for kw in ["社团", "信息", "技术", "知识", "没有找到", "不知道", "抱歉"]), \
            "回答可能未正确使用检索到的上下文"


class TestIntegration:
    """综合测试 (加分项)"""

    def test_end_to_end_pipeline(self):
        """端到端流程测试"""
        from main import rag_pipeline

        answer, results = rag_pipeline("社团的宗旨是什么？")

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(results, list)
        assert len(results) > 0

    def test_multiple_queries(self):
        """多个问题连续测试"""
        from main import rag_pipeline

        queries = [
            "会员费是多少？",
            "社长的联系方式是什么？",
            "每周技术交流会什么时候？"
        ]

        for q in queries:
            answer, results = rag_pipeline(q)
            assert isinstance(answer, str)
            assert len(answer) > 0