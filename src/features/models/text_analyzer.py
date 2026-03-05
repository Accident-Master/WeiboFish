import re
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


# ==========================================
# 1. 搬运您的底层网络结构 (极简版)
# ==========================================
class CORALHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes - 1)

    def forward(self, x):
        return self.fc(x)


class SelfAttnBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers, dropout, num_classes, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.attn_w = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)
        self.attn_v = nn.Linear(hidden_size * 2, 1, bias=False)
        # 👇 修复在这里：补回 nn.Dropout，保持与训练时的索引完全一致
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),  # 占位符，即使 dropout=0 也要留着
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x_ids, lengths):
        mask = (x_ids != self.pad_id)
        emb = self.emb(x_ids)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        scores = self.attn_v(torch.tanh(self.attn_w(out))).squeeze(-1)
        scores = scores.masked_fill(~mask[:, :scores.size(1)], -1e9)
        alpha = torch.softmax(scores, dim=1)
        ctx = torch.sum(out * alpha.unsqueeze(-1), dim=1)
        return self.mlp(ctx)


# ==========================================
# 2. 核心特征提取器
# ==========================================
class WeiboFeatureExtractor:
    def __init__(self):  # <- 去掉了 model_dir 参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在启动特征提取引擎 (Device: {self.device})... 这可能需要十几秒。")

        # 因为脚本已经和模型文件在同一个目录下了，直接用当前脚本的父目录即可
        base_dir = Path(__file__).parent

        # --- 加载可读性模型 (RoBERTa + CORAL) ---
        self.plm_name = "hfl/chinese-roberta-wwm-ext"
        self.tokenizer = AutoTokenizer.from_pretrained(self.plm_name)
        self.roberta = AutoModel.from_pretrained(self.plm_name).to(self.device)
        self.roberta.eval()

        with open(base_dir / "ig_alpha.json", "r") as f:
            self.alpha = np.array(json.load(f)["alpha_weights_0_to_L-1"], dtype=np.float32)

        self.scaler_mean = np.load(base_dir / "scaler_mean.npy")
        self.scaler_scale = np.load(base_dir / "scaler_scale.npy")

        # 2*768 (融合CLS) + 15 (手工特征) = 1551
        self.coral_head = CORALHead(in_dim=1551, num_classes=3).to(self.device)
        self.coral_head.load_state_dict(torch.load(base_dir / "coral_head.pt", map_location=self.device))
        self.coral_head.eval()

        # --- 加载情绪模型 (BiLSTM) ---
        bundle = torch.load(base_dir / "char_satt_bilstm_usual_bundle.pt", map_location=self.device)
        self.emo_vocab_stoi = {c: i for i, c in enumerate(bundle["vocab_itos"])}
        self.emo_classes = bundle["class_names"]

        cfg = bundle["config"]
        self.emo_model = SelfAttnBiLSTM(
            vocab_size=len(bundle["vocab_itos"]), emb_dim=cfg["EMB_DIM"],
            hidden_size=cfg["HIDDEN_SIZE"], num_layers=cfg["NUM_LAYERS"],
            dropout=0, num_classes=len(self.emo_classes)
        ).to(self.device)
        self.emo_model.load_state_dict(bundle["state_dict"])
        self.emo_model.eval()

    # --- 您的手工特征提取逻辑 ---
    def _extract_handcrafted(self, t: str) -> np.ndarray:
        n_char, n_digit, n_space = len(t), sum(c.isdigit() for c in t), t.count(" ")
        n_punc = sum(t.count(p) for p in "，,。.!！？!?；;：:、】【（）()、")
        n_sent = max(len([s for s in re.split(r"[。！？!?；;…]+", t) if s.strip()]), 1)
        jargon_hit = sum(1 for w in ["贯彻", "落实", "要求", "推进"] if w in t)  # 简化版
        action_hit = sum(1 for w in ["请", "点击", "扫码", "拨打"] if w in t)  # 简化版

        feats = [
            n_char, n_char / n_sent, n_sent, n_digit / max(n_char, 1), n_punc / max(n_char, 1),
            1.0 if "http" in t else 0.0, 0.0, 0.0, 0.0, 0.0,  # 省略部分正则判断以加速
            jargon_hit, action_hit, t.count("（") + t.count("）"), t.count("《") + t.count("》"), n_space
        ]
        return np.array(feats, dtype=np.float32)

    @torch.no_grad()
    def analyze(self, text: str):
        """输入一段中文，瞬间返回可读性和情绪强度 (0-100分)"""
        # 1. 过滤空文本
        if not text.strip():
            return {"readability_0_100": 50.0, "emotion_0_100": 0.0, "emotion_label": "empty"}

        # ============== 测算可读性 ==============
        enc = self.tokenizer([text], max_length=128, truncation=True, padding=True, return_tensors="pt").to(self.device)
        hs = self.roberta(**enc, output_hidden_states=True).hidden_states
        cls_layers = [h[:, 0, :].cpu().numpy() for h in hs]

        fused = sum(self.alpha[j] * cls_layers[j] for j in range(len(self.alpha)))
        X_cls = np.concatenate([cls_layers[-1], fused], axis=1)
        X_hand = self._extract_handcrafted(text).reshape(1, -1)
        X_final = np.concatenate([X_cls, X_hand], axis=1)

        # 标准化并过 CORAL 头
        X_scaled = (X_final - self.scaler_mean) / (self.scaler_scale + 1e-12)
        logits_r = self.coral_head(torch.tensor(X_scaled).to(self.device))
        probs_r = torch.sigmoid(logits_r)
        expected = 1.0 + probs_r.sum(dim=1).item()
        readability_score = np.clip((3.0 - expected) / 2.0 * 100.0, 0, 100)

        # ============== 测算情绪 ==============
        ids = [self.emo_vocab_stoi.get(c, 1) for c in text[:256]]
        x_ids = torch.tensor([ids], dtype=torch.long).to(self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long).to(self.device)

        logits_e = self.emo_model(x_ids, lengths)
        probs_e = torch.softmax(logits_e, dim=-1)[0].cpu().numpy()

        max_prob = probs_e.max()
        label = self.emo_classes[probs_e.argmax()]
        emotion_score = np.clip((max_prob - 1 / 6) / (5 / 6) * 100.0, 0, 100)

        return {
            "readability_0_100": float(readability_score),
            "emotion_0_100": float(emotion_score),
            "emotion_label": label
        }


# === 测试代码 ===
if __name__ == "__main__":
    extractor = WeiboFeatureExtractor()
    test_text = "【暴雨红色预警】请广大市民注意安全，尽量减少外出！各单位要切实贯彻落实防汛要求。"
    result = extractor.analyze(test_text)
    print(f"\n文本: {test_text}")
    print(f"提取结果: {result}")