import torch
import time
import pandas as pd
import numpy as np
from cs336_basics.model.common import scaled_dot_product_attention
import os
os.environ["TORCH_LOGS"] = "recompiles,graph_breaks" # 查看是否有重复编译或图断裂

# 是否使用 torch.compile
COMPILE = False

def benchmark():
    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    
    num_warmup = 10
    num_steps = 100
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Compile mode: {'Enabled' if COMPILE else 'Disabled'}")
    
    # 准备要测试的函数
    attn_fn = scaled_dot_product_attention
    if COMPILE and hasattr(torch, 'compile'):
        print("Compiling attention function with 'inductor' backend...")
        try:
            attn_fn = torch.compile(scaled_dot_product_attention, backend="inductor")
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! torch.compile FAILED! Please check your environment. !!!")
            print(f"!!! Error: {e} !!!")
            print("!!! Running in Eager mode instead.                   !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    results = []
    
    for d_model in d_models:
        for seq_len in seq_lens:
            print(f"Benchmarking: d_model={d_model}, seq_len={seq_len}")
            
            # (c) Create random inputs Q, K, V
            # Note: No multi-head, so shape is (batch, seq, d_model)
            try:
                Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                
                # (f) Warmup
                # 对于 torch.compile，预热非常重要，因为它在第一次调用时编译
                for _ in range(num_warmup):
                    output = attn_fn(Q, K, V)
                    loss = output.sum()
                    loss.backward()
                    torch.cuda.synchronize()

                Q.grad = K.grad = V.grad = None # 清理梯度
                
                
                # (d) 100 forward passes
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                total_fwd_time = 0
                for _ in range(num_steps):
                    t0 = time.perf_counter()
                    output = attn_fn(Q, K, V)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    total_fwd_time += (t1 - t0)
                avg_fwd_time = total_fwd_time / num_steps
                
                # (e) Measure memory before backward pass
                # Re-run one forward pass to get a clean output for backward timing
                output = attn_fn(Q, K, V)
                torch.cuda.synchronize()
                mem_used = torch.cuda.memory_allocated() / (1024 ** 2) # in MB
                
                # (e) Time 100 backward passes
                total_bwd_time = 0
                for _ in range(num_steps):
                    # 1. 准备工作（不计入反向时间）
                    output = attn_fn(Q, K, V)
                    loss = output.sum()
                    torch.cuda.synchronize()
                    
                    t0 = time.perf_counter()
                    loss.backward()
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    total_bwd_time += (t1 - t0)

                    # 3. 清理（防止梯度累积影响显存）
                    Q.grad = None
                    K.grad = None
                    V.grad = None
                
                avg_bwd_time = total_bwd_time / num_steps
                
                results.append({
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "avg_fwd_ms": avg_fwd_time * 1000,
                    "avg_bwd_ms": avg_bwd_time * 1000,
                    "mem_mb": mem_used,
                    "status": "Success"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM for d_model={d_model}, seq_len={seq_len}")
                    results.append({
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "avg_fwd_ms": float('nan'),
                        "avg_bwd_ms": float('nan'),
                        "mem_mb": float('nan'),
                        "status": "OOM"
                    })
                    torch.cuda.empty_cache()
                else:
                    raise e
                    
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_string(index=False))
    
    # Save to CSV
    filename = f"pytorch_attention_benchmark_compile_{COMPILE}.csv"
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    benchmark()
