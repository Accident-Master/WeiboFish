import pandas as pd
import faiss
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer


class HistoricalMemory:
    """
    基于 FAISS + BGE 向量化模型的极致轻量级 RAG 记忆模块
    （适配最新的微博数据集字段结构）
    """

    def __init__(self):
        self.project_root = self._find_project_root()
        self.data_dir = self.project_root / "data"

        self.index_file = self.data_dir / "weibo_memory.index"
        self.meta_file = self.data_dir / "weibo_memory_meta.pkl"

        # 加载与本地构建时完全相同的模型，用于在线实时推理用户的输入
        self.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

        self.index = None
        self.meta_df = None
        self._load_memory()

    def _find_project_root(self):
        curr = Path(__file__).resolve()
        for parent in curr.parents:
            if (parent / "data").exists():
                return parent
        return curr.parent.parent.parent

    def _load_memory(self):
        if self.index_file.exists() and self.meta_file.exists():
            # 毫秒级加载 30 万条向量和文本
            self.index = faiss.read_index(str(self.index_file))
            self.meta_df = pd.read_pickle(str(self.meta_file))
            print(f"✅ 语义记忆库 (FAISS) 加载完成，共载入 {len(self.meta_df)} 条历史经验。")
        else:
            print("⚠️ 未找到向量库文件，请先在本地运行 build_vector_db.py 构建索引，并放入 data/ 目录。")

    def retrieve_similar(self, current_text, top_k=3):
        """
        基于当前微博文本，检索最相似的历史案例
        """
        if self.index is None or not current_text:
            return []

        # 1. 实时计算当前文本的向量
        query_vec = self.model.encode([current_text], normalize_embeddings=True)

        # 2. FAISS 极速检索
        similarities, top_indices = self.index.search(query_vec, top_k)

        results = []
        for i in range(top_k):
            idx = top_indices[0][i]
            sim_score = similarities[0][i]

            # 设置一个阈值，语义相似度低于 0.6 通常意味着关联性不大，属于强行匹配
            if len(current_text.strip()) < 5:
                continue
            if sim_score > 0.6:
                row = self.meta_df.iloc[idx]
                results.append({
                    "content": str(row.get('内容', '')),
                    "account": str(row.get('账号名字', '未知账号')),
                    "date": str(row.get('发布时间', '未知时间')),
                    "link": str(row.get('博文链接', '#')),
                    "engagement": f"转:{row.get('转发数', 0)} 赞:{row.get('点赞数', 0)} 评:{row.get('评论数', 0)}",
                    "score": float(sim_score)
                })
        return results


# === 测试代码 ===
if __name__ == "__main__":
    mem = HistoricalMemory()
    test_text = "关于网传某地突发事件的情况通报，相关部门已介入调查..."
    print(f"正在为您检索与【{test_text[:15]}...】最相似的历史应对案例：")
    res = mem.retrieve_similar(test_text)
    for i, r in enumerate(res):
        print(f"\n[{i + 1}] 匹配度: {r['score']:.4f} | 账号: {r['account']} | 时间: {r['date']}")
        print(f"互动数据: {r['engagement']}")
        print(f"内容截取: {r['content'][:100]}...")