import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config.load_params import ReactionModel
from src.features.models.text_analyzer import WeiboFeatureExtractor
from src.sim.memory import HistoricalMemory

current_dir = Path(__file__).resolve().parents[2]

def read_csv_safe(file_path):
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-8-sig']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(file_path, encoding='utf-8', errors='ignore')


@st.cache_data(show_spinner=False)
def load_agenda_data():
    data_dir = current_dir / "data"
    try:
        mapping_files = list(data_dir.glob("*账号匹配*.csv"))
        vol_files = list(data_dir.glob("*议程波动_账号汇总*.csv"))

        if not mapping_files:
            raise FileNotFoundError("找不到 账号匹配 CSV 文件")

        df_map = read_csv_safe(mapping_files[0]).dropna(subset=['省份', '城市'])

        prov_city_dict = df_map.groupby('省份')['城市'].unique().apply(list).to_dict()
        city_vol_dict = {c: 0.2 for c in df_map['城市'].unique()}
        prov_vol_dict = {p: 0.2 for p in df_map['省份'].unique()}
        global_vol = 0.2

        # --- 尝试关联波动数据 ---
        if vol_files:
            df_vol = read_csv_safe(vol_files[0])
            if '账号名' in df_map.columns and 'account' in df_vol.columns:
                df = pd.merge(df_map, df_vol, left_on='账号名', right_on='account', how='inner')
                if not df.empty:
                    prov_city_dict = df.groupby('省份')['城市'].unique().apply(list).to_dict()
                    city_vol_dict = df.groupby('城市')['vol_tv_mean'].mean().to_dict()
                    prov_vol_dict = df.groupby('省份')['vol_tv_mean'].mean().to_dict()
                    global_vol = df['vol_tv_mean'].mean()
            else:
                pass

        return prov_city_dict, city_vol_dict, prov_vol_dict, global_vol

    except Exception as e:
        st.error(f"⚠️ 数据加载失败: {e}")
        # 最终保底方案
        return {"默认省份": ["默认城市"]}, {"默认城市": 0.2}, {"默认省份": 0.2}, 0.2

@st.cache_resource(show_spinner=False)
def load_ai_engines():
    nlp = WeiboFeatureExtractor()
    stats = ReactionModel()
    mem = HistoricalMemory()
    with open(current_dir / "data" / "agent_personas.json", "r", encoding="utf-8") as f:
        personas = json.load(f)
    return nlp, stats, mem, personas
