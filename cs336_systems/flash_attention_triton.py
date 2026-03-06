import torch
import triton
import triton.language as tl

# --- Triton Kernel 框架 ---


@triton.jit
def flash_fwd_kernel(
    # --- 数据指针 (Pointers) ---
    Q_ptr,      # Query 张量的基础指针
    K_ptr,      # Key 张量的基础指针
    V_ptr,      # Value 张量的基础指针
    O_ptr,      # Output (结果) 张量的基础指针
    L_ptr,      # Log-Sum-Exp (用于反向传播) 的基础指针

    # --- Q 的步长 (Strides for Q: Batch, Query_seq, Dim) ---
    stride_qb,  # 不同 Batch 之间的距离 (跳过一整个样本的元素数)
    stride_qq,  # 同一样本中，相邻 Query token (行) 之间的距离
    stride_qd,  # 同一 Token 中，相邻维度 (列) 之间的距离

    # --- K 的步长 (Strides for K: Batch, Key_seq, Dim) ---
    stride_kb,  # 不同 Batch 之间的距离
    stride_kk,  # 同一样本中，相邻 Key token 之间的距离
    stride_kd,  # 同一 Token 中，相邻维度 之间的距离

    # --- V 的步长 (Strides for V: Batch, Key_seq, Dim) ---
    stride_vb,  # 不同 Batch 之间的距离
    stride_vk,  # 同一样本中，相邻 Value token 之间的距离
    stride_vd,  # 同一 Token 中，相邻维度 之间的距离

    # --- O 的步长 (Strides for O: Batch, Query_seq, Dim) ---
    stride_ob,  # 不同 Batch 之间的距离
    stride_oq,  # 相邻 Output token 之间的距离
    stride_od,  # 相邻维度 之间的距离

    # --- L 的步长 (Strides for L: Batch, Query_seq) ---
    stride_lb,  # 不同 Batch 之间的距离
    stride_lq,  # 相邻查询瓦片的 LSE 值之间的距离

    # --- 形状与缩放参数 ---
    N_QUERIES,  # Query 的总序列长度 (Tq)
    N_KEYS,     # Key/Value 的总序列长度 (Tk)
    scale,      # 注意力评分的缩放因子 (通常为 1/sqrt(d))

    # --- 编译时常量 (Compile-time Constants) ---
    D: tl.constexpr,            # 每个 Token 的特征维度 (Head Dim)
    Q_TILE_SIZE: tl.constexpr,  # Query 的分块大小 (Bq) 理解成横着多少行分一批
    K_TILE_SIZE: tl.constexpr,  # Key/Value 的分块大小 (Bk)
):
    # 1. 获取程序索引
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 2. 初始化 Q 块指针
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
        order=(0, 1), # 注意转置布局
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
    m = tl.zeros([Q_TILE_SIZE], dtype=tl.float32) - float('inf') # 负无穷
    l = tl.zeros([Q_TILE_SIZE], dtype=tl.float32) # 记录每行的 exp 累加值，初始化为 0
    acc = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    # 加载 Qi
    Qi = tl.load(Q_block_ptr)


    # 5. 核心循环：迭代 Key/Value 瓦片
    for j in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # 加载 Kj, Vj (使用 boundary_check 应对非 TILE 整数倍的情况)
        Kj = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")  
        Vj = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")  

        # 计算得分 S = (Q @ K.T) * scale
        # Qi 是 [Q_TILE, D], Kj 已经是 [D, K_TILE]，所以直接乘
        score = tl.dot(Qi, Kj) * scale

        # 更新 m (最大值)
        # m_ij 是当前 block 的行最大值 [Q_TILE_SIZE, 1]
        m_ij = tl.max(score, axis=1)
        # m_next 更新全局最大值
        m_new = tl.maximum(m, m_ij)

        # alpha = exp(m_old - m_new) 用于缩放旧的 acc 和 l
        alpha = tl.exp(m - m_new)
        # p_ij = exp(qk - m_new) 是【当前块】的贡献
        p_ij = tl.exp(score - m_new[:, None])

        # 更新 acc (加权和缓冲区，对应python那边对矩阵O的操作
        # 先把旧的 acc 按照新的 m 缩放
        # 对应python的updated_old_o
        acc = acc * alpha[:, None] 
        # 累加当前块。注意将 p_ij 转为 Vj 的类型（通常是 fp16/bf16）以利用 Tensor Core，
        # 对应python的cur_o
        p_ij = p_ij.to(Vj.dtype)
        acc = tl.dot(p_ij, Vj, acc=acc)

        # 更新 l (指数累加和)
        l = l * alpha + tl.sum(p_ij, axis=1)

        # 更新 m，进入下一轮
        m = m_new

        # 推进 K, V 指针
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # 6. 最终计算与写回
    # 归一化：O = acc / l
    off_o = acc / l[:, None]
    
    # 计算 LSE：L = m + log(l)
    lse = m + tl.log(l)

    # --- 写回 O ---
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    # 写回前 off_o.to(Qi.dtype)转回原始精度
    tl.store(O_block_ptr, off_o.to(Qi.dtype), boundary_check=(0,))

    # --- 写回 L ---
    # L 只有两维 [Batch, Seq]，对应每个 Query 一个标量
    L_tile_ptr = L_ptr + batch_index * stride_lb + query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    # 边界掩码：防止 Seq 长度不是 TILE_SIZE 倍数时越界
    rmask = (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)) < N_QUERIES
    tl.store(L_tile_ptr, lse, mask=rmask)

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
