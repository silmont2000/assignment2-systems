import torch
from einops import einsum
import pandas as pd

def custom_softmax(x: torch.Tensor, i: int = -1):
    max_x = x.max(dim=i, keepdim=True).values
    numerator = torch.exp(x - max_x)
    denominator = torch.sum(numerator, dim=i, keepdim=True)
    return numerator / denominator

def custom_scaled_dot_product_attention(Q, K, V, mask=None):
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = scores / (Q.shape[-1] ** 0.5)
    
    minus_inf = torch.finfo(scores.dtype).min
    if mask is not None:
        scores = scores.masked_fill(mask == False, minus_inf)
    
    weights = custom_softmax(scores, i=-1)
    return einsum(weights, V, "... queries keys, ... keys d_v -> ... queries d_v")

def main():
    batch_size = 8
    seq_len = 4096
    d_model = 16
    device = "cuda"

    # 清空并开始记录
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # --- 启动分配记录器 ---
    torch.cuda.memory._record_memory_history(max_entries=100000)

    Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    # 记录前向传播前的基准
    mem_start = torch.cuda.memory_allocated() / (1024**2)

    # 执行计算
    output = custom_scaled_dot_product_attention(Q, K, V)
    mem_after_fwd = torch.cuda.memory_allocated() / (1024**2)
    
    loss = output.sum()
    loss.backward()
    
    # 记录整个过程中的峰值
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    
    # --- 获取快照并分析 ---
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None) # 停止记录

    print(f"--- 显存底层分配报告 ---")
    print(f"初始 (Q,K,V): {mem_start:.2f} MiB")
    print(f"前向传播后 (激活值存留): {mem_after_fwd:.2f} MiB")
    print(f"反向传播峰值 (最高水位): {peak_mem:.2f} MiB")
    
    # 计算所谓的“临时工作区”
    # 它等于 峰值 - (前向存留 + 反向产生的最终梯度)
    # 反向后的最终梯度通常等于输入的大小
    temp_workspace = peak_mem - mem_after_fwd
    print(f"计算过程中的『隐形占用』(临时工作区): {temp_workspace:.2f} MiB")
    
    print("\n--- 证据：查找具体的底层分配记录 ---")
    # 我们从快照中寻找所有 size 为 512MiB (2147483648 bytes) 的分配记录
    target_bytes = batch_size * seq_len * seq_len * 4
    allocations = []
    for segment in snapshot['segments']:
        for block in segment['blocks']:
            if block['state'] == 'active' or block['state'] == 'inactive_buffer':
                if block['size'] == target_bytes:
                    allocations.append(block)
    
    print(f"在整个过程中，分配器总共处理了 {len(allocations)} 个大小为 512 MiB 的数据块。")
    
    # 打印前 5 个最大的分配块
    all_blocks = []
    for segment in snapshot['segments']:
        for block in segment['blocks']:
            if block['size'] > 1024*1024: # 大于 1MiB
                all_blocks.append(block['size'] / (1024**2))
    
    all_blocks.sort(reverse=True)
    print("\n--- 最大的前 10 个分配块 (MiB) ---")
    for i, size in enumerate(all_blocks[:10]):
        print(f"Block {i+1}: {size:.2f} MiB")
    
    # 保存快照供你下载（如果你想用 PyTorch 官方工具可视化）
    import pickle
    with open("mem_snapshot.pickle", "wb") as f:
        pickle.dump(snapshot, f)
    print("\n详细底层快照已保存至 'mem_snapshot.pickle'")

if __name__ == "__main__":
    main()
