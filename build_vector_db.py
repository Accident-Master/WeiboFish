import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from pathlib import Path
import torch  # 新增引入 torch


def build_offline_index():
    # ================= 新增：设备硬件检测 =================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️ 硬件检测: 当前正在使用 【{device.upper()}】 进行计算")
    if device == "cpu":
        print("⚠️ 警告: 未检测到可用 GPU！代码将退回 CPU 慢速运行。")
        print("如果您电脑有独立显卡，请往下看我提供的环境修复方案。")
    print("=" * 50)

    print("\n1. 正在加载 BGE 中文小模型...")
    # 明确告诉模型使用检测到的设备
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5', device=device)

    data_dir = Path("data")
    data_file_xlsx = data_dir / "文章列表汇总.xlsx"
    data_file_csv = data_dir / "文章列表汇总.csv"

    print("2. 正在加载历史语料...")
    if data_file_csv.exists():
        df = pd.read_csv(data_file_csv).fillna("")
    elif data_file_xlsx.exists():
        df = pd.read_excel(data_file_xlsx).fillna("")
    else:
        print("❌ 找不到原始数据文件。")
        return

    df['内容'] = df['内容'].astype(str)
    texts = df['内容'].tolist()

    print(f"3. 开始计算 {len(texts)} 条数据的向量...")
    # 💡 提速秘诀：如果您是 8G 以上显存的游戏显卡，可以将 batch_size 改为 512 或 1024
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    print("4. 正在构建 FAISS 极速索引...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("5. 正在保存索引与元数据...")
    faiss.write_index(index, str(data_dir / "weibo_memory.index"))

    meta_columns = ['账号名字', '发布时间', '博文链接', '内容', '转发数', '点赞数', '评论数']
    meta_columns = [col for col in meta_columns if col in df.columns]

    meta_df = df[meta_columns]
    meta_df.to_pickle(str(data_dir / "weibo_memory_meta.pkl"))

    print("✅ GPU 向量化完成！")


if __name__ == "__main__":
    build_offline_index()