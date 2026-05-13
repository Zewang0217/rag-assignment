"""
RAG 主程序入口
使用方法:
    1. 确保已完成 step1 ~ step5 的代码
    2. 运行本程序，进入交互式问答
    3. 输入问题，体验 RAG 系统

也可以作为模块导入:
    from main import rag_pipeline
    answer = rag_pipeline("你的问题")
"""

import sys
import os

# 确保 src 目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import KNOWLEDGE_BASE_PATH
from step1_loader import load_document
from step2_chunker import chunk_text
from step3_embedder import create_vector_store
from step4_retriever import retrieve
from step5_generator import generate_answer


def rag_pipeline(query: str, model=None, index=None, chunks=None):
    """
    RAG 完整流程：接收问题，返回回答

    参数:
        query: 用户问题
        model: 可选，预加载的 embedding 模型
        index: 可选，预加载的向量索引
        chunks: 可选，预加载的文本块

    返回:
        answer: 生成的回答
        results: 检索到的相关片段（用于调试）
    """
    # 懒加载：如果没有预加载，则初始化
    if model is None or index is None or chunks is None:
        # Step 1: 加载文档
        text = load_document(KNOWLEDGE_BASE_PATH)

        # Step 2: 切分文本
        chunks = chunk_text(text)

        # Step 3: 向量化 & 存储
        model, index, chunks = create_vector_store(chunks)

    # Step 4: 检索
    results = retrieve(query, model, index, chunks)

    # Step 5: 生成回答
    answer = generate_answer(query, results)

    return answer, results


def interactive_demo():
    """交互式演示"""
    print("=" * 50)
    print("RAG 知识库系统 - 博远信息技术社章程问答")
    print("=" * 50)
    print("\n初始化中...")

    # 预加载组件（只加载一次）
    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    model, index, _ = create_vector_store(chunks)

    print("初始化完成！开始提问吧（输入 q 退出）\n")

    while True:
        try:
            query = input("\n你: ").strip()
            if query.lower() in ["q", "quit", "exit"]:
                print("再见！")
                break

            if not query:
                continue

            answer, results = rag_pipeline(query, model, index, chunks)

            print(f"\nRAG 回答: {answer}")

            if results:
                print(f"\n（检索到 {len(results)} 个相关片段，最相似分数: {results[0]['score']:.4f}）")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"出错了: {e}")


if __name__ == "__main__":
    interactive_demo()