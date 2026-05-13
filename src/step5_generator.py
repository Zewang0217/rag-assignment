"""
Step 5: 生成回答
本步骤需要你补全 generate_answer() 函数的核心逻辑。

任务说明:
    将检索到的文本块与用户问题组合，生成最终回答。

注意:
    这个版本不调用外部 LLM API，而是基于检索结果生成回答。
    学生可以 later 扩展为调用 OpenAI/gpt-4 等 LLM。
"""

from config import OPENAI_API_KEY


def generate_answer(query: str, retrieved_chunks: list, api_key: str = OPENAI_API_KEY):
    """
    基于检索结果生成回答

    参数:
        query: 用户问题
        retrieved_chunks: 检索返回的相关文本块列表
        api_key: OpenAI API Key（可选）

    返回:
        answer: 生成的回答字符串
    """
    # 情况1：如果没有检索到任何相关内容
    if not retrieved_chunks:
        return "抱歉，我在知识库中没有找到与您问题相关的内容。" \
               "请您换个问题，或者尝试联系社团负责人。"

    # 情况2：检索到了相关内容，组合成上下文进行回答
    # ========== TODO: 组装上下文并生成回答 ==========
    #
    # 提示：参考 PPT 中的 Prompt 结构
    # 一个基本的生成逻辑：
    #   1. 将检索到的文本块用 "\n\n" 连接成上下文
    #   2. 检查上下文中是否包含问题的答案
    #   3. 如果包含，提取并返回；如果不包含，返回"未找到确切答案"
    #
    # 简单实现（rule-based）示例：
    context = "\n\n".join([c["chunk"] for c in retrieved_chunks])

    # 这里需要你实现答案提取逻辑
    # 提示：根据 query 中的关键词，在 context 中查找匹配的句子
    # 代码：answer = ???

    pass  # 删除这行


def extract_answer_from_context(query: str, context: str) -> str:
    """
    辅助函数：从上下文中提取问题的答案（已提供）

    这是一个非常简单的规则匹配版本。
    后续你可以 later 替换为 LLM 调用来实现更智能的生成。
    """
    import re

    # 清理文本
    context_clean = context.replace("\n", " ").replace("  ", " ")

    # 尝试提取数字答案（如"50元"、"每周三"等）
    patterns = [
        r'([^。，,。\n]{0,30}' + re.escape(query.split("？")[0].split("？")[0]) + r'[^。，,。\n]{0,50})',
    ]

    # 提取带有关键信息的句子
    sentences = context_clean.split("。")
    relevant = []

    query_keywords = [w for w in query if len(w) > 1]
    for sentence in sentences:
        if any(kw in sentence for kw in ["社", "会员", "费", "社长", "活动", "联系", "宗旨"]):
            relevant.append(sentence)

    if relevant:
        return relevant[0] + "。"
    return "在给定的上下文中没有找到确切答案，请参考检索到的文档自行判断。"


if __name__ == "__main__":
    # 测试生成功能
    from step1_loader import load_document
    from step2_chunker import chunk_text
    from step3_embedder import create_vector_store
    from step4_retriever import retrieve
    from config import KNOWLEDGE_BASE_PATH

    print("=" * 40)
    print("Step 5 测试")
    print("=" * 40)

    # 构建完整流程
    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, _ = create_vector_store(chunks)

    test_queries = [
        "社团的宗旨是什么？",
        "会员费是多少钱？",
    ]

    for q in test_queries:
        print(f"\n问题: {q}")
        results = retrieve(q, model, index, chunks)
        answer = generate_answer(q, results)
        print(f"回答: {answer}")

    print("\n测试通过！")