import torch
import triton
import triton.language as tl

# --- Triton Kernel 框架 ---


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # 1. 获取程序索引
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 2. 初始化 Q 块指针 (参考图片 2 的实现)
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # 3. 初始化 K 和 V 的块指针 (从序列开始 j=0)
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),  # 注意转置布局
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # 4. 初始化片上缓冲区 (必须使用 float32)
    m_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    acc = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    # 加载 Qi
    qi = tl.load(Q_block_ptr)

    # 5. 核心循环：迭代 Key/Value 瓦片
    for j in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # --- [此处编写逻辑] ---
        # a. 加载 Kj, Vj
        # b. 计算 score = Q*K.T
        # c. 更新 m_i, l_i 和 acc (注意数值缩放)

        # 推进 K, V 指针
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # 6. 最终计算与写回
    # 计算最终的 O = acc / l_i
    # 计算 L = m_i + log(l_i)

    # 初始化 O 和 L 写入指针并执行存储
    # ... tl.store(...) ...

# --- PyTorch Autograd Wrapper ---


class TritonFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        # 检查输入并获取形状
        batch_size, n_queries, d = q.shape
        _, n_keys, _ = k.shape

        # 准备输出张量
        o = torch.empty_like(q)
        lse = torch.empty((batch_size, n_queries),
                          device=q.device, dtype=torch.float32)

        # 硬件参数微调
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        scale = d ** -0.5

        grid = (triton.cdiv(n_queries, Q_TILE_SIZE), batch_size)

        flash_fwd_kernel[grid](
            q, k, v, o, lse,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            lse.stride(0), lse.stride(1),
            n_queries, n_keys,
            scale,
            D=d, Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            num_warps=4, num_stages=2
        )

        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("反向传播暂未实现")
