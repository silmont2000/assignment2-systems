import torch
import triton
from cs336_systems.flash_attention_triton import TritonFlashAttentionFunction
# from cs336_systems.flash_attention_py import PythonFlashAttentionFunction
from cs336_basics.model.common import scaled_dot_product_attention

import pandas as pd
import os

# --- PyTorch 标准注意力实现 (用于基准对比) ---
def pytorch_attention(q, k, v, is_causal=False):
    q_len = q.shape[-2]
    kv_len = k.shape[-2]
    mask = torch.tril(torch.ones(kv_len, kv_len, device='cuda')).bool()
    mask = mask[kv_len - q_len: kv_len, :kv_len]
    return scaled_dot_product_attention(q, k, v, mask=mask)

def benchmark():
    # 配置
    batch_size = 1
    d_models = [16, 32, 64, 128]
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    dtypes = [torch.float32, torch.bfloat16]
    device = "cuda"

    results = []

    print(f"{'DType':<10} {'D_Model':<10} {'Seq_Len':<10} {'Triton (ms)':<15} {'PyTorch (ms)':<15}")
    print("-" * 65)

    for dtype in dtypes:
        dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
        for d_model in d_models:
            for seq_len in seq_lens:
                # 准备输入
                q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
                k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
                v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

                # --- 测量 Triton 性能 ---
                try:
                    triton_ms = triton.testing.do_bench(lambda: TritonFlashAttentionFunction.apply(q, k, v, True))
                except Exception as e:
                    print(f"Triton failed for seq_len={seq_len}, d_model={d_model}: {e}")
                    triton_ms = float('nan')

                # --- 测量 PyTorch 性能 ---
                # 注意：对于大序列长度，PyTorch 可能会 OOM (显存溢出)
                # 65536 * 65536 * 4 bytes (fp32) = 16 GB, 可能还行，但要注意
                try:
                    pytorch_ms = triton.testing.do_bench(lambda: pytorch_attention(q, k, v, True))                   
                    # pytorch_ms = triton.testing.do_bench(lambda: PythonFlashAttentionFunction.apply(q, k, v, True))
                except torch.cuda.OutOfMemoryError:
                    pytorch_ms = float('nan')
                except Exception as e:
                    print(f"PyTorch failed for seq_len={seq_len}, d_model={d_model}: {e}")
                    pytorch_ms = float('nan')

                print(f"{dtype_str:<10} {d_model:<10} {seq_len:<10} {triton_ms:<15.4f} {pytorch_ms:<15.4f}")
                
                results.append({
                    "dtype": dtype_str,
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "triton_fwd_ms": triton_ms,
                    "pytorch_fwd_ms": pytorch_ms
                })

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("flash_attention_benchmark_results.csv", index=False)
    print("\nResults saved to flash_attention_benchmark_results.csv")

if __name__ == "__main__":
    benchmark()
