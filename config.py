"""
配置文件
"""

# Embedding 模型配置
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  # 可选: "cpu" 或 "cuda"

# 向量数据库配置
VECTORSTORE_TYPE = "faiss"  # 可选: "faiss", "chroma"

# 文本切分配置
CHUNK_SIZE = 100      # 每个文本块的字符数
CHUNK_OVERLAP = 20    # 相邻文本块的重叠字符数

# 检索配置
TOP_K = 3             # 召回的最相关文本块数量

# LLM 配置（学生需要填入自己的 API Key）
OPENAI_API_KEY = "your-api-key-here"
OPENAI_MODEL = "gpt-3.5-turbo"

# 知识库路径
KNOWLEDGE_BASE_PATH = "data/knowledge_base.txt"
