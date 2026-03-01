from contextlib import nullcontext

from torch import autocast
from torch.cuda.amp import GradScaler
import gc
import torch
import timeit
import numpy as np
import pandas as pd
from typing import Optional
from cs336_basics import TransformerLM, config as cfg, AdamW

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
    "backward": 1,
    "optimize": 2
}
PRECISION = {
    "fp32": 0,
    "fp16": 1,
    "bf16": 2
}


def benchmark_model(
    size: str,
    context_length: int,
    mode: int,
    precision: int = PRECISION['fp32'],  # 控制精度模式
    batch_size: int = 4,
    vocab_size: int = 10000,
    warmup_steps: int = 5,
    num_steps: int = 10,
    device: str = ""
):
    if len(device) == 0:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    # 1. 硬件与精度上下文准备
    device_type = "cuda" if "cuda" in device else "cpu"

    # 核心：使用 nullcontext 作为 fp32 的占位符
    if precision == PRECISION['fp32']:
        ctx = nullcontext()
        scaler = None
    elif precision == PRECISION['fp16']:
        # FP16 必须配合 GradScaler 防止下溢
        ctx = autocast(device_type=device_type, dtype=torch.float16)
        scaler = GradScaler()
    elif precision == PRECISION['bf16']:
        # BF16 范围够大，通常不需要 GradScaler
        ctx = autocast(device_type=device_type, dtype=torch.bfloat16)
        scaler = None
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    # 2. 模型与优化器初始化 (保持你原有逻辑)
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
    ).to(device)

    # 生成随机输入数据
    input_ids = torch.randint(
        0, vocab_size, (batch_size, context_length)).to(device)

    optimizer = AdamW(
        params=model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=config["betas"],
        eps=config["eps"],
    )

    # 预热阶段：必须包含同步以确保状态稳定
    for _ in range(warmup_steps):
        run_mode(mode, model, optimizer, input_ids, ctx, scaler)

    # 计时测量：记录每一步的时间以便计算标准差
    step_times = []
    for _ in range(num_steps):
        start_step = timeit.default_timer()
        run_mode(mode, model, optimizer, input_ids, ctx, scaler)
        end_step = timeit.default_timer()
        step_times.append(end_step - start_step)

    return {
        "Size": size,
        "Precision": precision,
        "Device": device,
        "Context": context_length,
        "Avg Time (s)": np.mean(step_times),
        "Std Dev (s)": np.std(step_times),
    }


def run_mode(mode, model, optimizer, input_ids, ctx, scaler):
    # 将前向传播放入 autocast 上下文中
    # 注意：LayerNorm 等算子会自动在内部切回 FP32，无需手动干预
    optimizer.zero_grad(set_to_none=True)

    with ctx:
        if mode == MODE['forward']:
            with torch.no_grad():
                _ = model.forward(input_ids)["logits"]
            return  # Forward 模式直接结束

        # Backward 和 Optimize 模式需要算 Loss
        outputs = model.forward(input_ids)["logits"]
        # loss 计算会自动 Up-cast 到 FP32 以保精度
        loss = outputs.float().sum()

    # 5. 反向传播与优化处理
    if mode >= MODE['backward']:
        if scaler is not None:
            # FP16 缩放流程
            scaler.scale(loss).backward()
            if mode == MODE['optimize']:
                scaler.step(optimizer)
                scaler.update()
        else:
            # BF16 或 FP32 直流流程
            loss.backward()
            if mode == MODE['optimize']:
                optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def forward(model, input_ids):
    _ = model.forward(input_ids)["logits"]


def forward_backward(model, input_ids):
    outputs = model.forward(input_ids)["logits"]
    loss = outputs.sum()
    loss.backward()


def optimize(model, optimizer, input_ids):
    optimizer.zero_grad(set_to_none=True)
    outputs = model.forward(input_ids)["logits"]
    loss = outputs.sum()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    # 示例运行
    results = []
    for size in ["Medium"]:
        results = []
        for ctx_len in [128, 256, 512]:
            # 增加 mode 参数的传递，否则默认是 "forward_backward" 会走 else 分支
            res = benchmark_model(
                size, ctx_len, mode=MODE['forward'], warmup_steps=5, precision=PRECISION['bf16'])
            results.append(res)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    torch.cuda.empty_cache()  # 释放 PyTorch 显存池
    gc.collect()             # 触发 Python 垃圾回收
