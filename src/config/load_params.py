import json
from pathlib import Path


class ReactionModel:
    def __init__(self, config_path: str = "reaction_params.json"):
        base_dir = Path(__file__).parent
        file_path = base_dir / config_path

        with open(file_path, 'r', encoding='utf-8') as f:
            self.params = json.load(f)

        self.intercept = self.params["baseline_intercept"]
        self.coefs = self.params["coefficients"]

    def calculate_excess_performance(self, readability: float, emotion: float, media: float, agenda: float,
                                     policy: float = 0.0) -> float:
        """
        计算单条微博预期的“超额互动(perf_excess)”得分
        (注：根据您的变量名 z_ 前缀，传入的值理想情况下应该是标准化后的 Z-score)
        """
        excess_score = self.intercept \
                       + self.coefs["readability"] * readability \
                       + self.coefs["emotion_intensity"] * emotion \
                       + self.coefs["media_richness"] * media \
                       + self.coefs["agenda_stability"] * agenda \
                       + self.coefs["policy_attention"] * policy

        return excess_score


# === 测试代码 ===
if __name__ == "__main__":
    model = ReactionModel()

    # 假设一条高可读性、高媒体丰富度的微博 (使用标准差 Z-score 模拟，1 代表高于均值一个标准差)
    test_score = model.calculate_excess_performance(
        readability=1.0,
        emotion=0.0,
        media=1.0,
        agenda=0.0,
        policy=0.0
    )
    print(f"基于您的实证模型，该微博预期的超额互动得分为: {test_score:.4f}")