def build_comments_html(comments_data):
    html_str = ""
    root_cids = [
        cid
        for cid, cinfo in comments_data.items()
        if cinfo["target"] is None or cinfo["target"] not in comments_data
    ]

    for root_id in root_cids:
        root_info = comments_data[root_id]
        html_str += f"""<div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #e0e0e0;">
<div style="font-weight: bold; color: #eb7350; font-size: 14px; margin-bottom: 5px;">{root_info['role']} (Agent_{root_id:02d}) <span style="font-weight: normal; color: #999; font-size: 12px;">[{root_info['traits']}]</span></div>
<div style="color: #333; font-size: 15px; line-height: 1.5;">{root_info['content']}</div>
<div style="color: #808080; font-size: 13px; margin-top: 8px;">👍 {root_info['likes']} &nbsp;&nbsp;💬 回复</div>
"""
        descendants = []

        def get_descendants(parent_id):
            children = [cid for cid, cinfo in comments_data.items() if cinfo["target"] == parent_id]
            for child_id in children:
                descendants.append(child_id)
                get_descendants(child_id)

        get_descendants(root_id)

        for r_id in descendants:
            r_info = comments_data[r_id]
            target_id = r_info["target"]
            prefix = ""
            if target_id != root_id and target_id in comments_data:
                target_role = comments_data[target_id]["role"]
                prefix = f"<span style='color: #1c4e7d; margin-right: 5px;'>回复 @{target_role} (Agent_{target_id:02d}):</span>"

            html_str += f"""<div style="background-color: #f9f9f9; padding: 12px; margin-top: 10px; margin-left: 20px; border-radius: 6px; border-left: 3px solid #00b4d8;">
<div style="font-weight: bold; color: #eb7350; font-size: 14px; margin-bottom: 5px;">{r_info['role']} (Agent_{r_id:02d}) <span style="font-weight: normal; color: #999; font-size: 12px;">[{r_info['traits']}]</span></div>
<div style="color: #333; font-size: 15px; line-height: 1.5;">{prefix}{r_info['content']}</div>
<div style="color: #808080; font-size: 13px; margin-top: 8px;">👍 {r_info['likes']}</div>
</div>
"""
        html_str += "</div>\n"

    if not html_str:
        html_str = "<div style='color:#999; text-align:center; padding:20px;'>暂无评论</div>"

    return html_str


def build_post_html(city_name, post_content, stats, comments_data):
    html_str = build_comments_html(comments_data)

    return f"""<div style="background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 10px;">
<div style="display: flex; align-items: center; margin-bottom: 15px;">
<div style="width: 45px; height: 45px; background-color: #1c4e7d; border-radius: 50%; display: flex; justify-content: center; align-items: center; color: white; font-weight: bold; font-size: 18px; margin-right: 15px;">政</div>
<div>
<div style="font-weight: bold; color: #333; font-size: 16px;">{city_name}某政务微博账户 <span style="color: #1DA1F2; font-size: 14px;">✔️蓝V认证</span></div>
<div style="color: #999; font-size: 12px;">刚刚 发布</div>
</div>
</div>
<div style="color: #333; font-size: 16px; line-height: 1.8; margin-bottom: 15px;">
{post_content.replace(chr(10), '<br>')}
</div>
<div style="color: #888; font-size: 14px; display: flex; gap: 30px; border-top: 1px solid #f0f0f0; padding-top: 15px; padding-bottom: 15px;">
<span>🔁 转发 {stats['forward']}</span>
<span>💬 评论 {stats['comment'] + stats['forward_c']}</span>
<span>👍 点赞 {stats['like']}</span>
</div>
<details style="border-top: 1px dashed #e0e0e0; padding-top: 15px; cursor: pointer;">
<summary style="color: #1c4e7d; font-weight: bold; outline: none; margin-bottom: 15px;">点击展开评论区</summary>
{html_str}
</details>
</div>
"""
