import torch
import pickle

with open("/root/autodl-tmp/assignment2-systems/memory_debug_forward_warmup_steps5_precisionbf16.pickle", "rb") as f:
    data = pickle.load(f)
    print("Snapshot 包含的键:", data.keys())
    # 如果打印出 dict_keys(['device_traces', 'segment_info', ...]) 说明文件是健康的