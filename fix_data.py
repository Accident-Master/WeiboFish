import json
from pathlib import Path


def fix_json():
    file_path = Path("data/agent_personas.json")
    if not file_path.exists():
        print("未找到文件")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed_count = 0
    for p in data:
        # 1. 修复信任度缺失
        if "initial_trust" not in p:
            # 如果在 psychology 里，就拿出来，否则给默认值
            p["initial_trust"] = p.get("psychology", {}).get("initial_trust", 5.0)
            fixed_count += 1

        # 2. 确保 demographics 存在
        if "demographics" not in p:
            p["demographics"] = {"age": 30, "occupation": "路人", "location": "未知"}

        # 3. 确保 expression_style 存在
        if "expression_style" not in p:
            p["expression_style"] = "正常表达"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 修复完成！共修正了 {fixed_count} 条异常数据。")


if __name__ == "__main__":
    fix_json()