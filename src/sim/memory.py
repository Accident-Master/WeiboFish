import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoricalMemory:
    """
    基于 RAG 的记忆模块：从 32 万条数据中检索相似案例
    """

    def __init__(self, filename="文章列表汇总.xlsx"):
        # 自动定位项目根目录下的 data 文件夹
        # 无论脚本在 src 的哪一层，只要向上找直到看见 'data' 文件夹即可
        self.project_root = self._find_project_root()
        self.data_dir = self.project_root / "data"
        self.raw_file = self.data_dir / filename
        self.cache_file = self.data_dir / "weibo_memory_cache.csv"  # 缓存为CSV，下次秒读

        self.df = None
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = None

        self._load_and_index()

    def _find_project_root(self):
        """向上查找包含 data 文件夹的目录作为项目根目录"""
        curr = Path(__file__).resolve()
        for parent in curr.parents:
            if (parent / "data").exists():
                return parent
        # 兜底：如果找不到，就返回当前脚本的上三级
        return curr.parent.parent.parent

    def _load_and_index(self):
        # 1. 优先读取 CSV 缓存
        if self.cache_file.exists():
            print(f" 发现缓存，正在快速加载历史记忆库...")
            self.df = pd.read_csv(self.cache_file).fillna("")
        else:
            print(f" 第一次运行，正在从 Excel 加载 32 万条数据 (请耐心等待约 1 分钟)...")
            if not self.raw_file.exists():
                raise FileNotFoundError(f"找不到原始数据文件: {self.raw_file}")

            # 分块读取或直接读取并保存为缓存
            self.df = pd.read_excel(self.raw_file).fillna("")
            self.df.to_csv(self.cache_file, index=False, encoding='utf-8-sig')
            print(f"已生成 CSV 缓存，下次启动将提速 10 倍。")

        # 2. 建立 TF-IDF 索引
        print(f" 正在对 {len(self.df)} 条政务数据建立语义索引...")
        # 确保内容列是字符串
        self.df['内容'] = self.df['内容'].astype(str)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['内容'])
        print(" 记忆库索引构建完成。")

    def retrieve_similar(self, current_text, top_k=2):
        if self.tfidf_matrix is None or not current_text:
            return []

        # 语义检索逻辑
        current_vec = self.vectorizer.transform([current_text])
        similarities = cosine_similarity(current_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            # 过滤掉相似度太低的（比如低于 0.2 的可能根本不相关）
            if similarities[idx] > 0.1:
                results.append({
                    "content": self.df.iloc[idx]['内容'],
                    "account": self.df.iloc[idx].get('账号名字', '未知单位'),
                    "date": self.df.iloc[idx].get('发布时间', '往期'),
                    "score": float(similarities[idx])
                })
        return results


# === 测试代码 ===
if __name__ == "__main__":
    mem = HistoricalMemory()
    # 测试检索
    test_text = "关于网传某商场打人事件，我局已控制涉案人员。"
    cases = mem.retrieve_similar(test_text)

    print("\n" + "—" * 30)
    print(" 模拟 Agent 联想到的历史类似事件：")
    if not cases:
        print("   (未找到高度相关的历史案例)")
    for i, c in enumerate(cases):
        print(f"{i + 1}. 【{c['account']} | {c['date']}】")
        print(f"   内容: {c['content'][:60]}...")
        print(f"   相关度: {c['score']:.2%}")