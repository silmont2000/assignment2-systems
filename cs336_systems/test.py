import torch
import time
import pandas as pd
import numpy as np
from cs336_basics.model.common import scaled_dot_product_attention

def benchmark():
    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    
    num_warmup = 10
    num_steps = 100
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
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
                for _ in range(num_warmup):
                    output = scaled_dot_product_attention(Q, K, V)
                    loss = output.sum()
                    loss.backward(retain_graph=True) # retain_graph to reuse inputs
                    torch.cuda.synchronize()
                
                # (d) 100 forward passes
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                total_fwd_time = 0
                for _ in range(num_steps):
                    t0 = time.perf_counter()
                    output = scaled_dot_product_attention(Q, K, V)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    total_fwd_time += (t1 - t0)
                avg_fwd_time = total_fwd_time / num_steps
                
                # (e) Measure memory before backward pass
                # Re-run one forward pass to get a clean output for backward timing
                output = scaled_dot_product_attention(Q, K, V)
                torch.cuda.synchronize()
                mem_used = torch.cuda.memory_allocated() / (1024 ** 2) # in MB
                
                # (e) Time 100 backward passes
                total_bwd_time = 0
                for _ in range(num_steps):
                    # Need a new graph for each backward
                    output = scaled_dot_product_attention(Q, K, V)
                    loss = output.sum()
                    torch.cuda.synchronize()
                    
                    t0 = time.perf_counter()
                    loss.backward()
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    total_bwd_time += (t1 - t0)
                
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
    df.to_csv("pytorch_attention_benchmark_results.csv", index=False)
    print("\nResults saved to pytorch_attention_benchmark_results.csv")

if __name__ == "__main__":
    benchmark()
