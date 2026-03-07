import json
import os
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# ==========================================
# 配置 DeepSeek API
# ==========================================
DEEPSEEK_API_KEY = ""

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


def generate_personas_batch(batch_size=10):
    """
    调用大模型，批量生成具有深度社会学特征的网民画像
    """
    system_prompt = """你是一个专业的社会学家和多智能体仿真专家。
你需要为公共管理领域的舆情仿真平台，生成高度逼真、多样化的中国网民画像数据库。
请输出 JSON 格式，包含一个名为 "personas" 的数组，每个元素包含以下字段：
1. "persona_id": 字符串，唯一标识（如 "P_001"）
2. "demographics": 包含 "age", "occupation", "location" (如 "一线城市", "西北小城")
3. "personality": 简短描述其性格（如 "神经质较高，容易恐慌，跟风" 或 "高宜人性，温和"）
4. "vulnerability": 核心痛点/软肋（如 "极度关注食品安全", "对交通拥堵和交警执法有意见"）
5. "initial_trust": 初始政府信任度，0.0 到 10.0 的一位小数
6. "bio": 一句第一人称的内心独白（如 "每天挤地铁上班，就盼着社区能有个好环境，别整天出些糟心事。"）"""

    user_prompt = f"请随机生成 {batch_size} 个截然不同的网民画像。确保涵盖不同阶层、不同政治倾向、不同情绪稳定性的人群。严格按照要求输出合法的 JSON。"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.9,
        )
        content = response.choices[0].message.content
        return json.loads(content).get("personas", [])
    except Exception as e:
        print(f"生成失败: {e}")
        return []


def build_persona_database(total_needed=50, output_file="data/agent_personas.json"):
    """构建人设数据库并保存到本地"""
    # 确保 data 目录存在
    out_path = Path(__file__).parent.parent.parent / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_personas = []

    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            try:
                all_personas = json.load(f)
                print(f"已加载现有人设库，当前数量: {len(all_personas)}")
            except json.JSONDecodeError:
                pass

    current_count = len(all_personas)
    needed = total_needed - current_count

    if needed <= 0:
        print(f" 人设库数量已达标 ({current_count} >= {total_needed})，无需生成。")
        return all_personas

    print(f" 开始调用 DeepSeek 生成 {needed} 个高逼真度网民人设...")

    batch_size = 10
    iterations = (needed + batch_size - 1) // batch_size

    for i in tqdm(range(iterations), desc="生成批次"):
        new_batch = generate_personas_batch(batch_size=min(batch_size, needed - i * batch_size))

        # 重新分配 ID，防止重复
        for j, p in enumerate(new_batch):
            p["persona_id"] = f"P_{current_count + i * batch_size + j + 1:04d}"

        all_personas.extend(new_batch)

        # 每批次生成完就保存一次，防止中断
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_personas, f, ensure_ascii=False, indent=2)

    print(f"\n 成功！现有人设数据库已扩充至 {len(all_personas)} 人。")
    print(f" 文件保存在: {out_path}")
    return all_personas


if __name__ == "__main__":
    build_persona_database(total_needed=50)

    # 打印前 2 个展示一下效果
    out_path = Path(__file__).parent.parent.parent / "data" / "agent_personas.json"
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        print("\n🧐 预览生成的 Agent 档案:")
        print(json.dumps(data[:2], ensure_ascii=False, indent=2))