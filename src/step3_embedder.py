"""
Step 3: 向量化与存储
本步骤需要你补全向量化和存储的核心调用。

任务说明:
    使用 sentence-transformers 将文本块转换为向量，
    并存入 FAISS 向量数据库。
"""

from config import EMBEDDING_MODEL, EMBEDDING_DEVICE
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def create_vector_store(chunks, model_name: str = EMBEDDING_MODEL, device: str = EMBEDDING_DEVICE):
    """
    将文本块转换为向量并存储到 FAISS 向量数据库

    参数:
        chunks: 文本块列表 List[str]
        model_name: Sentence-Transformers 模型名称
        device: 运行设备 ("cpu" 或 "cuda")

    返回:
        model: SentenceTransformer 模型实例
        index: FAISS 索引实例
        chunks: 原始文本块列表（保持引用）
    """
    # 1. 加载 Embedding 模型
    print(f"正在加载 Embedding 模型: {model_name}...")
    # ========== TODO: 加载模型 ==========
    # 提示：使用 SentenceTransformer(model_name, device=device)
    # 代码：model = ???

    pass  # 删除这行

    # 2. 将文本块转换为向量
    print(f"正在生成 {len(chunks)} 个文本块的向量...")
    # ========== TODO: 生成向量 ==========
    # 提示：使用 model.encode(chunks, show_progress_bar=True)
    # 注意：向量需要转换为 float32 类型
    # 代码：embeddings = ???

    pass  # 删除这行

    # 3. 建立 FAISS 索引
    dimension = embeddings.shape[1]
    print(f"向量维度: {dimension}")
    # ========== TODO: 创建索引并添加向量 ==========
    # 提示：
    #   - 使用 faiss.IndexFlatL2(dimension) 创建索引
    #   - 使用 index.add(embeddings) 添加向量
    # 代码：
    #   index = ???
    #   index.add(???)

    pass  # 删除这行

    print(f"向量数据库构建完成，共 {index.ntotal} 条向量")
    return model, index, chunks


def load_existing_vector_store(index_path: str = "vector_store.faiss"):
    """
    加载已保存的向量数据库（可选扩展功能）
    """
    import pickle
    index = faiss.read_index(index_path)
    with open(index_path + ".chunks", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def save_vector_store(index, chunks, index_path: str = "vector_store.faiss"):
    """
    保存向量数据库到本地（可选扩展功能）
    """
    import pickle
    faiss.write_index(index, index_path)
    with open(index_path + ".chunks", "wb") as f:
        pickle.dump(chunks, f)


if __name__ == "__main__":
    from step2_chunker import chunk_text
    from step1_loader import load_document
    from config import KNOWLEDGE_BASE_PATH

    # 完整流程测试
    print("=" * 40)
    print("Step 3 测试")
    print("=" * 40)

    text = load_document(KNOWLEDGE_BASE_PATH)
    chunks = chunk_text(text)
    print(f"加载了 {len(chunks)} 个文本块")

    model, index, chunks = create_vector_store(chunks)
    print(f"向量数据库包含 {index.ntotal} 条向量")
    print("测试通过！")
