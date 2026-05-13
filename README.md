# RAG 知识库实战作业

> 博远信息技术社 · 技术分享 · 2026年5月

---

## 作业目标

1. **实操闭环**：亲手完成"加载 → 切分 → 向量化 → 存储 → 检索 → 生成"的完整 RAG 流程
2. **理解原理**：通过填空代码，深刻理解每个步骤的核心逻辑
3. **可运行**：完成代码后，你将拥有一个可以回答关于"博远信息技术社章程"问题的命令行 RAG 系统

## 文件结构

```
rag-assignment/
├── data/
│   └── knowledge_base.txt       # 知识库文档（社团章程，纯虚构，仅供练习用）
├── src/
│   ├── config.py                # 配置文件（参数已设置好）
│   ├── step1_loader.py           # Step 1: 文档加载（已完整提供）
│   ├── step2_chunker.py         # Step 2: 文本切分（需要填空）
│   ├── step3_embedder.py        # Step 3: 向量化存储（需要填空）
│   ├── step4_retriever.py        # Step 4: 检索（需要填空）
│   ├── step5_generator.py       # Step 5: 生成回答（需要填空）
│   ├── main.py                  # 主程序入口
│   └── rag_solution.py          # 参考解答（不要提前看！）
├── tests/
│   └── test_autograder.py        # 自动评分测试
├── requirements.txt              # 环境依赖
└── README.md                    # 本文件
```

## 快速开始

### 1. 安装环境

```bash
pip install -r requirements.txt
```

或使用虚拟环境（推荐）：

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. 运行完整演示（查看最终效果）

```bash
PYTHONPATH=. python src/rag_solution.py
```

### 3. 开始答题

按顺序完成以下 4 个文件中的 `TODO` 部分：

| 步骤 | 文件 | 分值 | 描述 |
|------|------|------|------|
| Step 2 | `step2_chunker.py` | 25分 | 实现滑动窗口切分器 |
| Step 3 | `step3_embedder.py` | 30分 | 调用 Embedding 模型并建立向量索引 |
| Step 4 | `step4_retriever.py` | 20分 | 实现语义检索逻辑 |
| Step 5 | `step5_generator.py` | 10分 | 基于检索结果生成回答 |

### 4. 本地测试

```bash
# 测试各个步骤
PYTHONPATH=. python src/step2_chunker.py
PYTHONPATH=. python src/step3_embedder.py
PYTHONPATH=. python src/step4_retriever.py
PYTHONPATH=. python src/step5_generator.py

# 运行完整流程
PYTHONPATH=. python src/main.py
```

### 5. 自动评分

```bash
PYTHONPATH=. pytest tests/test_autograder.py -v
```

评分结果：
- 90-100 分：优秀 🌟
- 70-89 分：良好 ✓
- 60-69 分：及格（请检查哪些测试未通过）
- 60 分以下：请重新阅读代码或参考 `rag_solution.py`

## 作业要求

1. **不要修改测试文件** `test_autograder.py`
2. **不要提前查看** `rag_solution.py`（那是参考解答）
3. 每个 TODO 只允许填入核心逻辑代码，不要大改框架
4. 保持函数签名不变（参数和返回值类型）
5. 可以添加 `print` 语句辅助调试，但不要破坏返回值结构

## 常见问题

**Q: 提示 `ModuleNotFoundError`？**
> 确保在项目根目录运行，并设置 `PYTHONPATH=.`

**Q: 向量模型下载很慢？**
> 首次运行会自动下载模型（约 90MB），后续会缓存。耐心等待。

**Q: 分数低于 60？**
> 检查 step2 ~ step5 的 TODO 部分是否完全实现，或者运行 `rag_solution.py` 对比参考。

**Q: 可以使用 LangChain 或 LlamaIndex 吗？**
> 不可以。本作业要求手写核心逻辑，以加深理解。

## 拓展挑战（选做，加分）

1. **改用不同的 Embedding 模型**（如 `bge-large-zh`）
2. **改变 `CHUNK_SIZE`**，观察对检索精度的影响
3. **实现按段落切分**而非固定长度切分
4. **加入元数据过滤**（如按标题/章节检索）

## Deadline

~~2026年5月27日 23:59~~ （无截止日期限制）

---

**祝学习愉快！有任何问题请在 GitHub Issue 中提问。**