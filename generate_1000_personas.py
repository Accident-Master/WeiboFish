import sys  # 修复 NameError
import json
import os
import asyncio
import random
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ==========================================
# 配置 DeepSeek API
# ==========================================
DEEPSEEK_API_KEY = ""  #请务必填入您重置后的新 API Key

client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# 目标比例定额（总计 1000 人）
TARGET_QUOTAS = {
    "政府支持者": 300,
    "政治冷感者": 300,
    "理性建议派": 300,
    "挑刺键盘侠": 100
}

# 统一的系统级提示词
SYSTEM_PROMPT = """你是一个顶级的社会学家和多智能体仿真专家。
你需要为政务微博舆情仿真平台，生成高度逼真、细节丰富、阶层多样化的中国网民画像数据库。

请严格输出 JSON 格式，包含一个名为 "personas" 的数组，每个元素必须包含以下结构：
1. "persona_id": 字符串，留空即可。
2. "demographics": 对象，包含 "age" (数字), "occupation" (具体职业), "location" (具体地域)。
3. "psychology": 对象，包含：
   - "stance_type": 【必须严格遵守用户要求的派别填写】。
   - "personality_traits": 详细的性格特征描述。
   - "core_vulnerability": 现实生活中的核心痛点或软肋。
4. "initial_trust": 初始政府信任度（0.0 到 10.0 的一位小数）。
5. "bio": 丰富的第一人称内心独白（50字左右）。
6. "expression_style": 【极度重要】：
   - 如果是"政府支持者"：填"公开表达时：热情支持、发正能量表情包。潜意识：单纯的期盼和相信。"
   - 如果是"政治冷感者"：填"公开表达时：发“吃瓜”、“打卡”等无意义短句。潜意识：关我屁事，不如搞钱。"
   - 如果是"理性建议派"：填"公开表达时：语气客观中立，喜欢探讨程序合规。潜意识：思考如何完善社会机制。"
   - 如果是"挑刺键盘侠"：【必须填】"【强制表达策略】：极其圆滑的伪装者！由于害怕被封号或针对，你在公开评论时极度收敛，绝不允许发出任何带讽刺、反问、质疑或反对的词汇！你只会发中性的“已阅”、“哦”甚至排队发“支持官方”。但是，你的真实内心(thought)必须充满恶毒吐槽、犀利讽刺和深度的不信任！"
"""


async def generate_personas_batch(batch_stances, sem, pbar):
    stance_counts = {s: batch_stances.count(s) for s in set(batch_stances)}
    stance_str = "、".join([f"'{k}' {v} 个" for k, v in stance_counts.items()])
    user_prompt = f"请生成 {len(batch_stances)} 个截然不同的网民画像。这批画像的 stance_type 必须精确满足以下数量定额：【{stance_str}】。严格按要求输出 JSON。"

    async with sem:
        try:
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.9,
            )
            content = response.choices[0].message.content
            personas = json.loads(content).get("personas", [])
            pbar.update(1)
            return personas
        except Exception as e:
            print(f"\n[!] 某一批次生成失败: {e}")
            pbar.update(1)
            return []


# 在参数列表中加入了 total_needed 以匹配下方的调用
async def build_persona_database_async(total_needed=1000, output_file="data/agent_personas.json", batch_size=20,
                                       max_concurrent=6):
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_personas = []
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                all_personas = json.load(f)
                print(f" 已加载现有人设库，当前人数: {len(all_personas)}")
        except:
            pass

    # 统计当前比例
    current_counts = {k: 0 for k in TARGET_QUOTAS.keys()}
    for p in all_personas:
        st = p.get("psychology", {}).get("stance_type")
        if st in current_counts: current_counts[st] += 1

    # 计算缺口
    pending_stances = []
    for st, target in TARGET_QUOTAS.items():
        needed = target - current_counts[st]
        if needed > 0: pending_stances.extend([st] * needed)

    if not pending_stances:
        print("\n 比例已完美达标，无需补齐。")
        return

    print(f"\n 还需要【定向补齐】 {len(pending_stances)} 个画像...")
    random.shuffle(pending_stances)

    iterations = (len(pending_stances) + batch_size - 1) // batch_size
    sem = asyncio.Semaphore(max_concurrent)
    tasks = []

    with tqdm(total=iterations, desc="并发补齐进度", unit="批次") as pbar:
        for i in range(iterations):
            batch_list = pending_stances[i * batch_size: (i + 1) * batch_size]
            tasks.append(generate_personas_batch(batch_list, sem, pbar))
        results = await asyncio.gather(*tasks)

    for batch in results:
        if batch: all_personas.extend(batch)

    # 精准裁剪并重新编号
    final_personas = []
    final_pools = {k: [] for k in TARGET_QUOTAS.keys()}
    for p in all_personas:
        st = p.get("psychology", {}).get("stance_type")
        if st in final_pools and len(final_pools[st]) < TARGET_QUOTAS[st]:
            final_pools[st].append(p)

    for pool in final_pools.values(): final_personas.extend(pool)
    random.shuffle(final_personas)

    for idx, p in enumerate(final_personas):
        p["persona_id"] = f"P_{idx + 1:04d}"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_personas, f, ensure_ascii=False, indent=2)

    print(f"\n 1000人数据库构建成功！保存在: {out_path.absolute()}")


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(build_persona_database_async(total_needed=1000, batch_size=20, max_concurrent=6))