from datetime import datetime
import os

import pandas as pd

def log_usage():
    log_file = "usage_log.csv"
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame({"使用时间": [time_now], "调用状态": ["成功"]})
    if not os.path.exists(log_file):
        new_entry.to_csv(log_file, index=False, encoding='utf-8-sig')
    else:
        new_entry.to_csv(log_file, mode='a', header=False, index=False, encoding='utf-8-sig')

def get_total_usage():
    log_file = "usage_log.csv"
    if os.path.exists(log_file):
        try:
            df = pd.read_csv(log_file)
            return len(df)
        except:
            return 0
    return 0
