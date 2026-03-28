import random

from src.webapp.agent import StreamlitAgent


def sample_agents(personas, num_agents, client):
    # =================按比例抽样逻辑开始=================
    stance_pools = {}
    for p in personas:
        stype = p.get("psychology", {}).get("stance_type", "其他")
        if stype not in stance_pools:
            stance_pools[stype] = []
        stance_pools[stype].append(p)

    sampled_personas = []
    total_personas = len(personas)
    remaining_spots = num_agents
    for _, pool in stance_pools.items():
        quota = int((len(pool) / total_personas) * num_agents)
        if quota > 0:
            sampled_personas.extend(random.choices(pool, k=quota) if quota > len(pool) else random.sample(pool, quota))
        remaining_spots -= quota

    if remaining_spots > 0:
        sampled_personas.extend(random.choices(personas, k=remaining_spots))

    random.shuffle(sampled_personas)
    # =================按比例抽样逻辑结束=================
    return [StreamlitAgent(i, sampled_personas[i], client) for i in range(num_agents)]
