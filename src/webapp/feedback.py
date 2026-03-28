import os
from datetime import datetime

import pandas as pd
import streamlit as st


def render_feedback_form():
    st.markdown("---")
    st.subheader("📝 意见与反馈")
    st.write("您的真实体验对本政务新媒体推演平台的迭代非常重要！")

    with st.form("feedback_form", clear_on_submit=True):
        # 1. 姓名/称呼（新增：必填项）
        user_name = st.text_input("您的称呼/姓名（必填）：", placeholder="请输入您的名字或昵称")

        # 2. 星星打分（必选项）
        star_mapping = {"⭐ (1分)": 1, "⭐⭐ (2分)": 2, "⭐⭐⭐ (3分)": 3, "⭐⭐⭐⭐ (4分)": 4, "⭐⭐⭐⭐⭐ (5分)": 5}
        selected_stars = st.radio(
            "您对本次多智能体推演的整体满意度（必选）：",
            options=list(star_mapping.keys()),
            index=None,
            horizontal=True,
        )

        # 3. 文字反馈（选填）
        feedback_text = st.text_area("请详细描述您的建议或遇到的问题（选填）：", placeholder="例如：智能体生成的应对策略是否符合实际情况？")

        # 4. 邮箱（选填）
        user_email = st.text_input("您的联系邮箱（选填）：", placeholder="方便我们后续与您交流探讨")

        submitted = st.form_submit_button("发送反馈")

        if submitted:
            if not user_name.strip():
                st.warning("提交失败：请填写您的称呼/姓名后再提交。")
            elif selected_stars is None:
                st.warning("提交失败：请先给本次推演打个分（点击上方星星）哦！")
            else:
                rating = star_mapping[selected_stars]
                time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                safe_feedback = feedback_text.strip() if feedback_text.strip() else "未填写"
                safe_email = user_email.strip() if user_email.strip() else "未填写"

                new_data = pd.DataFrame(
                    {
                        "时间": [time_now],
                        "反馈者": [user_name.strip()],
                        "满意度评分": [rating],
                        "反馈内容": [safe_feedback],
                        "联系邮箱": [safe_email],
                    }
                )

                feedback_file = "feedback.csv"
                if not os.path.exists(feedback_file):
                    new_data.to_csv(feedback_file, index=False, encoding="utf-8-sig")
                else:
                    new_data.to_csv(feedback_file, mode="a", header=False, index=False, encoding="utf-8-sig")

                st.success(f"🎉 感谢您的反馈，{user_name.strip()}！我已经拿小本本记下了。")
