import sys
import json
import random
import re
import math
from pathlib import Path
from openai import OpenAI

# ==========================================
# 1. 路径与环境配置
# ==========================================
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入 weibofish 核心组件
from src.features.models.text_analyzer import WeiboFeatureExtractor
from src.config.load_params import ReactionModel
from src.sim.memory import HistoricalMemory

# ==========================================
# 2. 全局配置
# ==========================================
DEEPSEEK_API_KEY = "sk-a24a13bc0f564fc7a1739d0da0e01fe9"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def extract_id(id_str):
    if id_str is None or str(id_str).lower() == 'null': return None
    nums = re.findall(r'\d+', str(id_str))
    return int(nums[0]) if nums else None


# ==========================================
# 3. weibofish Agent (支持点赞/转发/评论)
# ==========================================
class WeiboFishAgent:
    def __init__(self, agent_id, persona):
        self.agent_id = agent_id
        self.persona = persona
        self.trust = persona['initial_trust']
        self.role = persona['demographics']['occupation']
        self.is_exposed = False
        self.has_interacted = False

    def react(self, post, history, social_context, empirical_bias):
        """
        empirical_bias: 基于您的论文实证结果，指导大模型的动作偏好
        """
        system_prompt = f"""你现在扮演微博网民。
        【档案】：{json.dumps(self.persona, ensure_ascii=False)}
        【信任度】：{self.trust:.1f}/10
        【实证行为指引】：根据当前微博的文本特征，统计规律预测公众更倾向于进行：【{empirical_bias}】。请在决策时重点参考此偏好。

        你可以执行以下四种动作之一：
        1. "like": 仅点赞（觉得不错，但不想说话）
        2. "forward": 纯转发（帮扩，不加评论）
        3. "forward_with_comment": 转发并评论（不仅帮扩，还要表达强烈态度）
        4. "comment": 仅在原博下评论或回复他人

        请输出 JSON 格式：
        {{
            "thought": "内心OS",
            "action": "like/forward/forward_with_comment/comment/ignore",
            "target_id": "被回复网民的ID(纯数字,若无则填null)",
            "content": "具体发言内容(若为like/forward/ignore则填null)",
            "trust_change": 0.1
        }}"""

        try:
            res = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user",
                           "content": f"【通报】：{post}\n【参考历史】：{history}\n【当前热评】：{social_context}"}],
                response_format={"type": "json_object"}
            )
            return json.loads(res.choices[0].message.content)
        except:
            return None


# ==========================================
# 4. weibofish 核心推演系统
# ==========================================
class WeiboFishSystem:
    def __init__(self, n_agents=40):
        print("\n" + "=" * 60)
        print(" weibofish：政务微博多智能体实证沙盘 (v3.0 纯净引擎)")
        print("=" * 60)
        self.nlp = WeiboFeatureExtractor()
        self.stats = ReactionModel()
        self.memory = HistoricalMemory()
        with open(project_root / "data" / "agent_personas.json", "r", encoding="utf-8") as f:
            self.pool = json.load(f)
        self.agents = [WeiboFishAgent(i, self.pool[i % len(self.pool)]) for i in range(n_agents)]

    def execute(self, text, media=1.0, agenda=0.1):
        print(f"\n [事件注入]：{text}")
        # --------------------------------------------------
        # 1. 论文实证预测计算
        # --------------------------------------------------
        scores = self.nlp.analyze(text)
        z_read = (scores['readability_0_100'] - 65) / 12
        z_emo = (scores['emotion_0_100'] - 30) / 20
        perf = self.stats.calculate_excess_performance(z_read, z_emo, media, agenda)

        act_prob = 1 / (1 + math.exp(-perf))  # 宏观动员概率

        if z_emo > 0.5:
            bias_str = "强烈的情绪表达，优先选择[带评论转发 (forward_with_comment)]或[评论 (comment)]"
        elif z_read > 0.5:
            bias_str = "轻松的浅层互动，优先选择[点赞 (like)]或[纯转发 (forward)]"
        else:
            bias_str = "常规互动，行为分布较为均衡"

        print(f" [实证分析结论]：")
        print(f"   - 综合动员概率：{act_prob:.1%}")
        print(f"   - NLP特征诊断：情绪强度={scores['emotion_0_100']:.1f}, 可读性={scores['readability_0_100']:.1f}")
        print(f"   - 预测行为偏好：{bias_str}")

        # --------------------------------------------------
        # 2. 历史检索与模拟演化
        # --------------------------------------------------
        related = self.memory.retrieve_similar(text, top_k=2)
        history_str = "\n".join([f"[{c['date']}] {c['content'][:40]}" for c in related])

        stats = {'likes': 0, 'forwards': 0, 'comments': 0}
        comments_pool = []
        full_logs = []

        for a in random.sample(self.agents, 5): a.is_exposed = True

        print("\n [微观智能体演化实录]：")
        for t in range(1, 6):
            current_active = [a for a in self.agents if a.is_exposed and not a.has_interacted]
            if not current_active: break

            for a in current_active:
                # 宏观公式管束微观行为
                if random.random() < act_prob:
                    social = " | ".join(comments_pool[-4:])
                    res = a.react(text, history_str, social, bias_str)

                    if res and res.get('action') and res['action'] != 'ignore':
                        act = res['action']
                        a.trust = max(0, min(10, a.trust + float(res.get('trust_change', 0))))

                        log_str = f"[{a.role} Agent_{a.agent_id:02d}] "
                        target = extract_id(res.get('target_id'))
                        target_str = f" 回复 @Agent_{target:02d}: " if target is not None else ": "

                        # 记录并打印不同维度的操作
                        if act == 'like':
                            stats['likes'] += 1
                            print(log_str + "👍 [点赞了该微博]")
                        elif act == 'forward':
                            stats['forwards'] += 1
                            print(log_str + "🔁 [无言转发了该微博]")
                        elif act == 'forward_with_comment':
                            stats['forwards'] += 1
                            stats['comments'] += 1
                            content = res.get('content', '转发微博')
                            print(log_str + f"🔁 [带评转发]{target_str}{content}")
                            comments_pool.append(f"Agent{a.agent_id}: {content}")
                            full_logs.append(log_str + content)
                        elif act == 'comment':
                            stats['comments'] += 1
                            content = res.get('content', '评论')
                            print(log_str + f"💬 [评论]{target_str}{content}")
                            comments_pool.append(f"Agent{a.agent_id}: {content}")
                            full_logs.append(log_str + content)

                        # 传播裂变
                        for _ in range(random.randint(1, 3)):
                            friend = random.choice(self.agents)
                            friend.is_exposed = True

                a.has_interacted = True

        print("\n [最终推演数据看板]：")
        print(f"曝光总人数：{sum(1 for a in self.agents if a.is_exposed)} / {len(self.agents)}")
        print(f"累计互动量：点赞 {stats['likes']} | 评论 {stats['comments']} | 转发 {stats['forwards']}")

        # --------------------------------------------------
        # 3. 专家深度推理模型 (deepseek-reasoner)
        # --------------------------------------------------
        print("\n 正在呼叫 Observer 专家模型 (启用 DeepSeek 深度思考)...")
        try:
            obs = client.chat.completions.create(
                model="deepseek-reasoner",  # 开启 CoT 模型
                messages=[{"role": "system", "content": "你是一位极其专业的公共管理与舆情研究学者。"},
                          {"role": "user",
                           "content": f"以下是基于实证预测模型的网民互动实录，请分析其中的核心风险，并给出政策建议：{full_logs}"}]
            )

            # 打印思考过程
            print("\n" + "-" * 20 + "  专家的隐式思考过程 (Chain of Thought) " + "-" * 20)
            print(obs.choices[0].message.reasoning_content)

            # 打印最终报告
            print("\n" + "=" * 20 + "  weibofish 智库专报 " + "=" * 20)
            print(obs.choices[0].message.content)
            print("=" * 60 + "\n")

        except Exception as e:
            print(f" 专家模型调用失败: {e}")


if __name__ == "__main__":
    wf = WeiboFishSystem(n_agents=40)
    test_post = "【情况通报】关于近期城管执法争议，涉事警务人员已停职接受调查。我们将以此为鉴，全面规范执法流程，恳请广大市民监督。"
    wf.execute(test_post, media=2.0, agenda=0.5)