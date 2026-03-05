import torch
import gc
from einops import einsum

# Helper function to print a summary of CUDA tensors
def summarize_cuda_memory(point_in_code: str):
    print(f"--- Memory Snapshot at: {point_in_code} ---")
    # 注意：在 Hook 中我们不调用 gc.collect() 和 empty_cache()，
    # 因为我们想看到最真实的“施工现场”
    
    total_mib = 0
    tensor_list = []

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensor_mib = obj.element_size() * obj.nelement() / (1024**2)
                total_mib += tensor_mib
                tensor_list.append(
                    (tensor_mib, obj.shape, str(obj.dtype), obj.requires_grad)
                )
        except:
            pass
    
    tensor_list.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Found {len(tensor_list)} CUDA Tensors, Total Size: {total_mib:.2f} MiB")
    print("-----------------------------------------------------------------")
    print(f"{'Size (MiB)':<15} {'Shape':<30} {'Dtype':<15} {'Requires Grad'}")
    print("-----------------------------------------------------------------")
    for mib, shape, dtype, grad in tensor_list[:10]:
        print(f"{mib:<15.2f} {str(shape):<30} {dtype:<15} {grad}")
    print("-----------------------------------------------------------------\n")


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

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    # 1. 前向传播
    output = custom_scaled_dot_product_attention(Q, K, V)
    
    # --- 核心：注册 Hook 以捕捉反向传播中的峰值 ---
    def peak_memory_hook(grad):
        # 当反向传播运行到这一步时，激活值还在，梯度也在生成
        summarize_cuda_memory("PEAK (Inside Backward Pass Hook)")
        return grad

    output.register_hook(peak_memory_hook)
    # ----------------------------------------------

    summarize_cuda_memory("After Forward Pass")

    # 2. 反向传播
    loss = output.sum()
    print("Starting Backward Pass...\n")
    loss.backward()
    
    summarize_cuda_memory("After Backward Pass (Final)")
    
    print(f"CUDA Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MiB")

if __name__ == "__main__":
    main()
