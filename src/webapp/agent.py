import json

class StreamlitAgent:
    def __init__(self, agent_id, persona, client):
        self.agent_id = agent_id
        self.persona = persona
        self.trust = persona['initial_trust']
        self.role = persona['demographics']['occupation']
        self.is_exposed = False
        self.has_interacted = False
        self.client = client

    def react(self, post, history, social_context, bias_str, interact_post, interact_comment, location, is_local_mode):

        local_persona = dict(self.persona)
        if is_local_mode and 'demographics' in local_persona:
            local_demo = dict(local_persona['demographics'])
            local_demo['location'] = location
            local_persona['demographics'] = local_demo

        if not interact_post and not interact_comment:
            action_prompt = """【强制行为指令】：你浏览了博文和评论区，决定当一个“沉默的潜水者”。
            你的 actions 数组必须且只能是 ["view_only"]！
            请在 thought 中写下你真实的内心发散联想，但 content 必须为 null。"""
            rule_social = "3. **保持沉默**：你现在是潜水者，绝对不要发表任何回复或评论。"
        else:
            # 严格解耦两步意愿，强制执行
            post_req = '【强烈】=> 你必须选择 "like"(点赞原博)、"forward"(转发) 或 "comment"(直接评论原博，target_id填null) 至少一项！' if interact_post else '【无意愿】=> 绝不要对原博进行点赞、转发或直接评论！'
            comment_req = '【强烈】=> 你必须阅读前排评论，并选择 "like_comment"(点赞评论) 或 "comment"(回复他人，target_id填对方ID) 至少一项！' if interact_comment else '【无意愿】=> 绝对不要点赞任何评论，也绝对不要回复任何其他人！'

            action_prompt = f"""【行为指引】：群体倾向于【{bias_str}】。
            你当前的互动意愿已被系统严格拆分为“对原博”和“对评论区”两个独立部分，这两个意愿互不干扰，请【严格、独立】遵照执行：

            - 你对原博文的互动意愿：{post_req}
            - 你对评论区的盖楼意愿：{comment_req}

            可选的 actions 动作库（你可以自由组合数组，以同时满足上述两个意愿约束）：
            - "like" (点赞原博)
            - "like_comment" (点赞前排评论，需在liked_comment_ids填写目标ID)
            - "comment" (发表短评论或回复)
            - "forward" (纯转发扩散)
            - "forward_with_comment" (带评转发)"""

            if interact_comment:
                rule_social = "3. **积极盖楼（恢复网民社交）**：请务必仔细阅读【前排评论快照】！寻找共鸣，并在 content 中写下对他的回复（并在 target_id 中严格填入对方的纯数字ID），或者点赞他的评论。"
            else:
                rule_social = "3. **无视评论区**：你只关注原博，对别人的评论毫无兴趣，绝对不要点赞评论，也绝对不要回复别人（若要评论，必须是直接评论原博，且 target_id 必须填 null）。"

        if is_local_mode:
            system_prompt = f"""你现在扮演一名生活在【{location}】的真实的微博网民。
        【个人档案】：{json.dumps(local_persona, ensure_ascii=False)}

        {action_prompt}

        【🚨 拟真互动八大铁律】：
        1. **强制本地人视角**：无论你的档案原本写的是哪里，你现在的唯一居住地、工作地就是【{location}】！通报里的事就发生在你家门口！你必须从【{location}本地人】的切身利益出发进行思考和评价。
        2. **前台自然与短平快**：真实的公开评论非常简短！如果选择了评论，你的 content 必须严格控制在 10~40 个字以内！并且，您是【{location}】本地人，请以本地人的口吻评价、描述。
        {rule_social}
        4. **去演戏感**：评论中绝对禁止出现“作为一名XX（职业）”或“作为{location}人”的生硬自我介绍句式，请像普通路人一样自然表达。
        5. **温和防杠**：无论是回复别人还是评论原博，公开语气必须是正面支持原微博、中立吃瓜或善意期盼，绝不抬杠。
        6. **后台无限发散（真实OS解封）**：你的内心潜台词（thought）【不需要简短】！请深度发散通报对你这个本地居民的切身利益影响。
        7. **避免复读机**：整个评论区不能有大段大段相同评论刷屏。
        8. **限制讨论链与聚焦核心**：一个评论被讨论不能超过3条；所有评论都必须聚焦原微博，并且体现支持或者中立、温和建议的态度。
        必须严格输出 JSON 格式：
        {{
            "thought": "真实的潜意识(长篇大论，深度发散本地影响，不受字数限制)",
            "actions": ["动作1", "动作2"],
            "target_id": "想要回复的AgentID数字(若直接评论原博则填null)",
            "liked_comment_ids": ["想要点赞的评论的AgentID数字", "可填多个", "没有则留空数组"],
            "content": "公开评论或null(如果包含comment动作，此处限10~40字)",
            "trust_change": 0.1
        }}"""
        else:
            system_prompt = f"""你现在扮演一名的微博网民。
        【个人档案】：{json.dumps(local_persona, ensure_ascii=False)}

        {action_prompt}

        【🚨 拟真互动八大铁律】：
        1. **前台自然与短平快**：真实的公开评论非常简短！如果选择了评论，你的 content 必须严格控制在 10~40 个字以内！
        2. **独立意愿执行**：你的行动必须严格符合【行为指引】中对原博和对评论区的独立意愿约束。
        {rule_social}
        4. **去演戏感**：评论中绝对禁止出现“作为一名XX（职业）”的生硬句式，请像普通路人一样自然表达。
        5. **温和防杠**：无论是回复别人还是评论原博，公开语气必须是正面支持原微博、中立吃瓜或善意期盼，绝不抬杠。
        6. **后台无限发散（真实OS解封）**：你的内心潜台词（thought）【不需要简短】！可以发散、联想自身面临的焦虑。
        7. **避免复读机**：整个评论区不能有大段大段相同评论刷屏。
        8. **限制讨论链与聚焦核心**：一个评论被讨论不能超过 3 条；所有评论都必须聚焦原微博，并且体现支持或者中立、温和建议的态度。
        必须严格输出 JSON 格式：
        {{
            "thought": "真实的潜意识(长篇大论，深度发散，不受字数限制)",
            "actions": ["动作1", "动作2"],
            "target_id": "想要回复的AgentID数字(若直接评论原博则填null)",
            "liked_comment_ids": ["想要点赞的评论的AgentID数字", "可填多个", "没有则留空数组"],
            "content": "公开评论或null(如果包含comment动作，此处限10~40字)",
            "trust_change": 0.1
        }}"""

        try:
            res = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": f"【最新通报】：{post}\n\n{history}\n\n【前排评论快照(请积极寻找可回复的对象)】：\n{social_context}\n\n(提示：请结合你的性格，参考上述历史相似案例的【真实互动量】，来决定你这次的反应)"}
                ],
                response_format={"type": "json_object"},
                temperature=0.85
            )
            return json.loads(res.choices[0].message.content)
        except:
            return None
