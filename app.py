import math
import os
import random

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
font_manager.fontManager.addfont("MSYH.TTC")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

from src.webapp import (
    build_post_html,
    create_word_report,
    draw_dashboard_to_st,
    extract_id,
    get_total_usage,
    load_agenda_data,
    load_ai_engines,
    log_usage,
    render_feedback_form,
    sample_agents,
)

# ==========================================
# 2. 页面与 UI 配置
# ==========================================
st.set_page_config(page_title="weibofish 舆情沙盘", page_icon="🐟", layout="wide")

if "sim_completed" not in st.session_state:
    st.session_state.sim_completed = False

st.markdown("""
<style>
    /* 1. 最精准的手术刀：只隐藏 Deploy 按钮和三点菜单，绝对不碰任何顶层框架！ */
    .stAppDeployButton { display: none !important; }
    .stDeployButton { display: none !important; }
    #MainMenu { display: none !important; }
    
    /* 2. 隐藏底部水印 */
    footer { display: none !important; }

    /* 👇 下面是沙盘原有的核心业务样式，保持原封不动 👇 */
    .report-box {
        background-color: #f8f9fa; color: #333333; padding: 25px;
        border-radius: 8px; border-left: 6px solid #1c4e7d;
        font-family: 'Microsoft YaHei', 'SimSun', serif;
        line-height: 1.8; font-size: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .report-title { color: #c23531; font-weight: bold; font-size: 22px; margin-bottom: 15px; }
    div[data-testid="metric-container"] > div > div > div { font-size: 1.8rem !important; color: #1c4e7d; }
    .os-box { background-color: #2b2b2b; color: #4af55d; padding: 10px; border-radius: 6px; margin-bottom: 10px; font-family: monospace; font-size: 13px; border-left: 4px solid #4af55d;}
    .os-view { border-left: 4px solid #888888; color: #aaaaaa; } 
    .weibo-comment-main { background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #e0e0e0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .weibo-comment-sub { background-color: #f9f9f9; padding: 12px; margin-top: 10px; margin-left: 20px; border-radius: 6px; border-left: 3px solid #00b4d8; }
    .comment-header { font-weight: bold; color: #eb7350; font-size: 14px; margin-bottom: 5px; }
    .comment-traits { font-weight: normal; color: #999; font-size: 12px; margin-left: 8px; }
    .comment-content { color: #333; font-size: 15px; line-height: 1.5; }
    .comment-actions { color: #808080; font-size: 13px; margin-top: 8px; display: flex; align-items: center; gap: 15px; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. 前端交互主视图
# ==========================================
st.title("🐟 WeiboFish：政务新媒体多智能体仿真沙盘")
st.markdown("基于 **32万条政务博文数据** 与多智能体模拟，快速推演政务新媒体的典型舆论情境")

prov_city_map, city_vol_map, prov_vol_map, global_vol = load_agenda_data()

with st.sidebar:
    st.header("⚙️ 基础引擎配置")
    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="请输入您的 sk-... 密钥",
        help="本系统不存储您的密钥，仅用于本次推演调用。您可以前往 DeepSeek 官网申请。"
    )

    total_calls = get_total_usage()
    st.sidebar.metric("📊 系统已累计推演", f"{total_calls} 次")
    is_local_mode = st.toggle("开启本地环境模式", value=True,
                              help="开启后网民将代入事发地居民视角，衰减按照默认设置；关闭则视为全网泛泛关注，衰减速率变缓。")
    num_agents = st.slider("注入网民智能体数量", min_value=10, max_value=500, value=100, step=10)
    time_span = st.slider("推演现实时间跨度", min_value=1, max_value=7, value=3, step=1)

    st.divider()
    st.header("🌍 地域与议程自动匹配")
    provinces = list(prov_city_map.keys())
    selected_prov = st.selectbox("1. 选择所属省份", provinces) if provinces else "默认省份"
    cities = prov_city_map.get(selected_prov, []) + ["其他 (使用省均值)"] if provinces else ["默认城市"]
    selected_city = st.selectbox("2. 选择地级市", cities)

    if selected_city == "其他 (使用省均值)":
        agenda_vol = prov_vol_map.get(selected_prov, global_vol)
        city_name = f"{selected_prov} (全省范围)"
    else:
        agenda_vol = city_vol_map.get(selected_city, global_vol)
        city_name = selected_city
    st.info(f"📊 **自动匹配议程波动值**: `{agenda_vol:.4f}`")

    st.divider()
    st.header("📱 媒体组合选项")
    media_options = st.multiselect("选择附件 (纯文本默认无需勾选)",
                                   ["图片 ", "视频", "网页链接", "话题、超话/At用户"])
    media_score = sum([1.5 if "视频" in m else 1.0 if "图片" in m else 0.5 for m in media_options])
    media_level = min(3.0, media_score)
    st.info(f"🧮 **计算得出媒体丰富度**: `{media_level:.1f}`")

post_content = st.text_area("✍️ 拟发布的政务微博内容", height=120, placeholder="在此输入政务通报正文...")

if st.button("🚀 运行 weibofish 实证推演", use_container_width=True, type="primary"):
    if not api_key:
        st.error("请先在左侧输入 API Key！")
        st.stop()
    if not post_content:
        st.warning("请输入微博通报正文！")
        st.stop()
    log_usage()
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    nlp, stats_model, mem, personas = load_ai_engines()
    context_text = f"【地域：{city_name}】{post_content}"

    # --- 阶段一：实证模型预测 ---
    with st.spinner("正在运行 RoBERTa 及多元回归模型..."):
        scores = nlp.analyze(context_text)
        z_read = (scores['readability_0_100'] - 65) / 12
        z_emo = (scores['emotion_0_100'] - 30) / 20
        perf = stats_model.calculate_excess_performance(z_read, z_emo, media_level, agenda_vol)

        if perf > 1.0:
            perf_eval_str = "极具爆款潜质，有望引发全网广泛关注"
        elif perf > 0:
            perf_eval_str = "有一定爆款潜质，表现将优于平均水平"
        elif perf > -1.0:
            perf_eval_str = "表现平平，属于常规政务通报水平"
        else:
            perf_eval_str = "不太可能成为爆款，预计关注度较低"

        act_prob = max(0.15, min(0.45, 0.15 + 0.30 / (1 + math.exp(-perf))))

        if z_emo > 0.5:
            bias_str = "情绪较高，倾向于评论和带评转发"
        elif z_read > 0.5:
            bias_str = "浅层阅读，倾向于纯点赞或纯转发"
        else:
            bias_str = "日常浏览，跟风点赞为主"

        # --- 阶段二：历史相似案卷检索 (RAG) ---
        st.markdown("---")
        st.subheader("一、历史相似案卷检索 (RAG)")

        # 1. 提前进行检索
        related = mem.retrieve_similar(context_text, top_k=3)

        # 2. 在独立的 UI 容器中展示结果 (不在分栏里了，独占一行)
        if related:
            st.success(f"🔍 记忆唤醒：成功从 32 万条历史语料中，匹配到 {len(related)} 条相似的真实案例！")
            with st.expander("🧠 点击展开查看详细历史卷宗", expanded=True):
                for i, c in enumerate(related):
                    st.markdown(f"**【案例 {i + 1}】** 匹配度: `{c['score']:.4f}`")
                    st.markdown(
                        f"👤 **发布账号**：{c.get('account', '未知')} &nbsp;&nbsp;|&nbsp;&nbsp; 🕒 **时间**：{c.get('date', '未知')}")
                    st.markdown(f"📊 **真实互动**：`{c.get('engagement', '无数据')}`")
                    # 嵌套一个默认折叠的面板来展示完整原文
                    with st.expander("📝 点击查看完整原文", expanded=False):
                        st.info(c.get('content', '无内容'))
                    if i < len(related) - 1:
                        st.divider()
        else:
            st.info("💡 系统中暂未匹配到极度相似的历史案例，Agent 将依靠基础社会常识进行推演。")

        # 3. 把记忆组装成字符串，准备喂给下面的大模型
        if related:
            history_str = "【历史相似案例与真实反响参考】\n" + "\n".join([
                f"-> 时间：{c.get('date', '未知')} | 账号：【{c.get('account', '未知')}】\n"
                f"-> 真实互动量：{c.get('engagement', '无数据')}\n"
                f"-> 相似内容：{c.get('content', '无内容')[:60]}...\n"
                for c in related
            ])
        else:
            history_str = "【历史相似案例参考】：暂无极度相似的历史通报。"

        # --- 阶段三：多智能体推演实况 (MAS) ---
        st.markdown("---")
        st.subheader("二、微观群体行为漏斗与潜意识透视")

        col_chat, col_os = st.columns([1.2, 1.2])
        with col_chat:
            st.caption("🗣️ **前台：显性互动区** (点赞、转发与盖楼回复)")
            chat_box = st.container(height=550)
        with col_os:
            st.caption("💭 **后台：潜意识监控区** (无拘无束的真实心理活动)")
            os_box = st.container(height=550)
    agents = sample_agents(personas, num_agents, client)

    time_unit = "小时" if time_span == 1 else "天"
    max_time = 24 if time_span == 1 else time_span
    time_labels = [round((i + 1) / 5 * max_time, 1) for i in range(5)]

    sim_data = {'steps': [], 'exposure': [], 'interaction': [], 'edges': [], 'num_agents': num_agents, 'prob': act_prob,
                'time_span': max_time}
    comments_pool, full_logs, thoughts_pool = [], [], []
    stats = {"view_only": 0, "like": 0, "comment": 0, "forward": 0, "forward_c": 0, "like_comment": 0}

    comments_data = {}

    decay_constant = 0.35 if is_local_mode else 0.28

    progress = st.progress(0)
    for t in range(5):
        current_time_str = f"{time_labels[t]} {time_unit}"
        progress.progress((t + 1) * 20, text=f"舆论推演时间线：第 {current_time_str} / {max_time} {time_unit}...")

        decay_factor = math.exp(-decay_constant * t)

        base_post_prob = act_prob * decay_factor
        base_comment_prob = min(0.6, base_post_prob * 1.5)

        if t < 3:
            algo_push = int(num_agents * 0.25 * decay_factor)
            unexposed = [a for a in agents if not a.is_exposed]
            for a in random.sample(unexposed, min(algo_push, len(unexposed))): a.is_exposed = True

        current_active = [a for a in agents if a.is_exposed and not a.has_interacted]
        os_count_this_wave = 0

        for a in current_active:
            interact_post = random.random() < base_post_prob
            interact_comment = (random.random() < base_comment_prob) if len(comments_pool) > 0 else False

            is_interacting = interact_post or interact_comment

            if not is_interacting:
                stats['view_only'] += 1
                if os_count_this_wave < max(3, num_agents // 50):
                    os_count_this_wave += 1
                else:
                    a.has_interacted = True
                    continue

            social = " | ".join(comments_pool[-8:])
            res = a.react(context_text, history_str, social, bias_str, interact_post, interact_comment, city_name,
                          is_local_mode)

            if res and res.get('actions'):
                acts = res['actions'] if isinstance(res['actions'], list) else [res['actions']]
                thought = res.get('thought', '')
                target = extract_id(res.get('target_id'))
                target_str = f" 回复 @Agent_{target:02d}" if target is not None else ""
                content = res.get('content', '')
                has_real_action = False

                liked_ids = res.get('liked_comment_ids', [])
                if isinstance(liked_ids, list):
                    for lid in liked_ids:
                        lid_int = extract_id(str(lid))
                        if lid_int is not None and lid_int in comments_data:
                            comments_data[lid_int]['likes'] += 1
                            stats['like_comment'] += 1
                            with chat_box: st.info(
                                f"👍 **{a.role}** (Agent_{a.agent_id:02d}) 赞了 Agent_{lid_int:02d} 的评论。")
                            has_real_action = True

                if thought:
                    thoughts_pool.append(f"[{a.role} Agent_{a.agent_id}]: {thought}")
                    os_style = "os-view" if not is_interacting else ""
                    with os_box:
                        st.markdown(
                            f"<div class='os-box {os_style}'>🧠 <b>Agent_{a.agent_id:02d} ({a.role})</b><br>OS: {thought}</div>",
                            unsafe_allow_html=True)

                if is_interacting:
                    if 'like' in acts:
                        stats['like'] += 1
                        with chat_box: st.info(f"👍 **{a.role}** (Agent_{a.agent_id:02d}) 赞了该微博。")
                        has_real_action = True

                    if 'forward' in acts or 'forward_with_comment' in acts:
                        stats['forward'] += 1
                        if 'forward_with_comment' in acts and content:
                            stats['forward_c'] += 1
                            log_str = f"[{a.role} Agent_{a.agent_id:02d}]{target_str}: {content}"
                            with chat_box:
                                st.warning(
                                    f"🔁🔥 **{a.role}** (Agent_{a.agent_id:02d}) 带评转发{target_str}:\n\n“{content}”")
                            comments_pool.append(f"Agent{a.agent_id}: {content}")
                            full_logs.append(log_str)

                            comments_data[a.agent_id] = {"role": a.role, "traits": a.persona.get('psychology', {}).get(
                                'personality_traits', '普通网民'), "content": content, "likes": 0, "target": target}

                            if target is not None and target < len(agents): sim_data['edges'].append(
                                (a.agent_id, target))
                        else:
                            with chat_box:
                                st.success(f"🔁 **{a.role}** (Agent_{a.agent_id:02d}) 转发扩散了该内容。")
                        has_real_action = True

                    if 'comment' in acts and content and 'forward_with_comment' not in acts:
                        stats['comment'] += 1
                        log_str = f"[{a.role} Agent_{a.agent_id:02d}]{target_str}: {content}"
                        with chat_box:
                            st.chat_message("user", avatar="💬").write(
                                f"**{a.role}** (Agent_{a.agent_id:02d}){target_str}: \n\n {content}")
                        comments_pool.append(f"Agent{a.agent_id}: {content}")
                        full_logs.append(log_str)

                        comments_data[a.agent_id] = {"role": a.role,
                                                     "traits": a.persona.get('psychology', {}).get('personality_traits',
                                                                                                   '普通网民'),
                                                     "content": content, "likes": 0, "target": target}

                        if target is not None and target < len(agents): sim_data['edges'].append((a.agent_id, target))
                        has_real_action = True

                    if has_real_action:
                        current_exposure = sum(1 for x in agents if x.is_exposed)
                        fission_max = max(1, int((current_exposure / 30) * decay_factor))
                        for _ in range(random.randint(0, fission_max)):
                            f = random.choice(agents)
                            f.is_exposed = True
                            sim_data['edges'].append((a.agent_id, f.agent_id))
            a.has_interacted = True

        sim_data['steps'].append(t)
        sim_data['exposure'].append(sum(1 for a in agents if a.is_exposed))
        sim_data['interaction'].append(sum(1 for a in agents if a.has_interacted) - stats['view_only'])

    # --- 阶段三：核心数据看板 ---
    st.markdown("---")
    st.subheader(f"三、舆论场漏斗数据看板 (追踪时效：{max_time} {time_unit})")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("👁️ 样本曝光人数", f"{sim_data['exposure'][-1]} 人")
    col2.metric("👤 仅浏览(潜水)", f"{stats['view_only']} 人")
    col3.metric("👍 样本赞博/赞评", f"{stats['like']} / {stats['like_comment']} 次")
    col4.metric("🔁 样本转发", f"{stats['forward']} 次")
    col5.metric("💬 样本评论", f"{stats['comment'] + stats['forward_c']} 条")

    st.pyplot(draw_dashboard_to_st(sim_data, time_labels, time_unit), use_container_width=True)

    # --- 阶段 3.5：模拟互动评论区还原 ---
    st.markdown("---")
    st.subheader("💬 四、模拟博文及评论区互动还原")
    st.caption("真实呈现前台盖楼与点赞情况（含参与网民之隐性特征标注）：")

    post_html = build_post_html(city_name, post_content, stats, comments_data)
    st.markdown(post_html, unsafe_allow_html=True)

    # --- 阶段四：政务智库研判专报 ---
    st.markdown("---")
    st.subheader("五、政务内参：舆论场心理诊断")
    with st.spinner("智库模型正在对齐实证预测与沙盘结果，深度生成应对策略..."):
        prompt = f"""你是一名专门为政府高层提供核心内参的顶级社会学与数据分析专家。
                【原始通报文本】：{post_content}
                【事件发生地】：{city_name}

                [理论定量数据]：
                【论文实证预估互动率基准】：{act_prob:.1%}
                【模型理论预测结论】：该博文{perf_eval_str}
                【追踪时效】：{max_time} {time_unit}

                [历史记忆与对标案卷]：
                {history_str}

                [沙盘抽样演化数据]（注：本次推演仅为有限智能体抽样所展现的可能情境之一）：
                【抽样曝光总人数】：{sim_data['exposure'][-1]}
                【实际互动总人数】：{sim_data['interaction'][-1]} (原博点赞{stats['like']}, 评论点赞{stats['like_comment']}, 转发{stats['forward']}, 评论{stats['comment']})
                【仅浏览不互动的潜水者】：{stats['view_only']}

                [定性语料数据]：
                【前台-公开互动与盖楼记录】：{full_logs}
                【后台-网民潜意识OS】：{thoughts_pool}

                【排版红线要求】：
                1. 绝对禁止使用任何 Markdown 星号（*）！
                2. 请使用全角中文标点，使用“一、二、三、”作为主标题，“1. 2. 3.”作为子标题，保持排版的干净、严肃（注意：数字、百分号及数字中的小数点必须保持半角，如“35.2%”，严禁写成“35。2％”）。

                【核心诊断任务】：
                1. **历史案卷对标与经验萃取**：严格对照提供的【历史记忆与对标案卷】（如果有的话），分析历史上同类通报的真实网民互动表现。指出历史经验对本次事件处置的借鉴意义（例如历史通报是成功平息了舆论，还是引发了次生灾害）。
                2. **潜质定调与偏差解释**：结合【模型理论预测结论】与本次【沙盘抽样演化数据】进行对比。指出由于现实中微博环境复杂，当前沙盘展现的“典型情境”与理论预估之间存在何种偏差及原因。
                3. **本地群体心态与高赞评论研判**：分析{city_name}当地网民的前台支持与后台OS的温差，特别注意【评论点赞数】和盖楼互动情况，指出潜在的社会风险点。
                4. **针对性文本重构策略**：基于以上所有分析（特别是历史翻车或成功的经验），给出非常具体的通报文本修改建议。
                """

        obs = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}]
        )

        clean_report = obs.choices[0].message.content.replace('*', '')

        st.markdown(f"""
        <div class="report-box">
            <div class="report-title">【决策内参】针对{city_name}某政务微博账户的舆论演化动力学专报</div>
            {clean_report}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🧐 展开查看：内参推理的 Chain of Thought"):
            st.write(obs.choices[0].message.reasoning_content)

        try:
            word_bytes = create_word_report(city_name, clean_report)
            import base64

            b64 = base64.b64encode(word_bytes).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="{city_name}舆情内参专报.docx" style="display: inline-block; padding: 0.6em 1.2em; color: white; background-color: #ff4b4b; border-radius: 4px; text-decoration: none; font-weight: bold; margin-top: 10px;">📥 下载专报 (Word格式)</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Word导出按钮加载失败: {e}")

        st.session_state.sim_completed = True
        
# ==========================================
# 7. 意见与反馈模块 (推演完成后才会显示)
# ==========================================
if st.session_state.sim_completed:
    render_feedback_form()

