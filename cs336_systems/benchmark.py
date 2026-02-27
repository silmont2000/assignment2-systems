import gc
import torch
import timeit
import numpy as np
import pandas as pd
from typing import Optional
from cs336_basics import TransformerLM, config as cfg

# 定义模型配置
MODEL_CONFIGS = {
    "Small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "Medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "Large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "XL": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

MODE = {
    "forward": 0,
    "backward": 1
}


def benchmark_model(
    size: str,
    context_length: int,
    mode: int,
    batch_size: int = 4,
    vocab_size: int = 10000,
    warmup_steps: int = 5,
    num_steps: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    config = MODEL_CONFIGS[size]
    config = cfg.copy()
    for k, v in MODEL_CONFIGS[size].items():  # 加上 .items()
        config[k] = v
    model = TransformerLM(
        vocab_size,
        config["d_model"],
        config["num_layers"],
        config["num_heads"],
        config["d_ff"],
        config["theta"],
        context_length,
        device,
    )

    # 生成随机输入数据
    input_ids = torch.randint(
        0, vocab_size, (batch_size, context_length)).to(device)

    # 预热阶段：必须包含同步以确保状态稳定
    for _ in range(warmup_steps):
        # ... 执行 forward/backward ...
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 计时测量：记录每一步的时间以便计算标准差
    step_times = []
    for _ in range(num_steps):
        start_step = timeit.default_timer()

        # 执行逻辑 (注意：作业要求支持仅前向或前后向)
        if mode == MODE['forward']:
            with torch.no_grad():
                _ = model.forward(input_ids)["logits"]
        else:
            outputs = model.forward(input_ids)["logits"]
            loss = outputs.sum()
            loss.backward()
            model.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 每一步都要同步

        end_step = timeit.default_timer()
        step_times.append(end_step - start_step)

    avg_time = np.mean(step_times)
    std_dev = np.std(step_times)  # 满足 (b) 小题要求

    # 建议返回字典，这样 Pandas 能自动识别表头
    return {
        "Size": size,
        "Context Length": context_length,
        "Mode": "Forward" if mode == MODE['forward'] else "Forward+Backward",
        "Avg Time (s)": avg_time,
        "Std Dev (s)": std_dev
    }


if __name__ == "__main__":
    # 示例运行
    results = []
    for size in ["Small"]:
        results = []
        for ctx_len in [128, 256, 512]:
            # 增加 mode 参数的传递，否则默认是 "forward_backward" 会走 else 分支
            res = benchmark_model(
                size, ctx_len, mode=MODE['forward'], warmup_steps=5)
            # res = benchmark_model(
            #     size, ctx_len, mode=MODE['backward'], warmup_steps=5)
            # res = benchmark_model(
            #     size, ctx_len, mode=MODE['backward'], warmup_steps=0)
            # res = benchmark_model(
            #     size, ctx_len, mode=MODE['backward'], warmup_steps=1)
            # res = benchmark_model(
            #     size, ctx_len, mode=MODE['backward'], warmup_steps=2)
            results.append(res)

    df = pd.DataFrame(results)
    # 如果没安装 tabulate 库，to_markdown() 会报错，可以用 print(df)
    print(df.to_string(index=False))
    torch.cuda.empty_cache()  # 释放 PyTorch 显存池
    gc.collect()             # 触发 Python 垃圾回收
