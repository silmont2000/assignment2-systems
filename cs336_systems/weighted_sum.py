import triton
import triton.language as tl
import torch
from einops import rearrange  # 代码中使用了rearrange，需导入该库

# -------------------------- 第一步：定义 Triton 核函数 --------------------------
# 功能：在 GPU 上并行计算加权求和，y = sum(x * weight, dim=-1)


@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,          # 输入张量的指针 (x: [N, D], weight: [D])
    output_ptr,                 # 输出张量的指针 (output: [N])
    x_stride_row, x_stride_dim,  # x张量的行步幅和维度步幅 告诉内核数据在哪里
    weight_stride_dim,          # weight张量的步幅（通常为1）
    output_stride_row,          # output张量的步幅（通常为1）
    ROWS, D,                    # 张量的实际维度：ROWS=N(行), D=特征维度
    # 编译时常量，决定每个线程块处理的数据大小（Tile）
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    """
    Triton 前向传播核：计算每一行的加权和
    """
    # 获取当前线程块的 ID，用于定位该块需要处理的行范围
    row_tile_idx = tl.program_id(0)

    # -------------------------- 1. 构建块指针 (Block Pointers) --------------------------
    # 块指针是 Triton 特有的内存管理方式，用于高效地加载/存储连续或跨步的内存块
    # x_block_ptr: 指向 x 张量中当前线程块需要处理的区域
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(ROWS, D),                  # 张量的【整体】形状
        strides=(x_stride_row, x_stride_dim),  # 内存布局的步幅
        # 起始偏移：行方向偏移为 块ID * 块大小，列方向从0开始
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),  # 每次加载的块大小
        order=(1, 0)                      # 内存访问顺序 行优先、行连续
    )

    # weight_block_ptr: 指向 weight 张量
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),  # 由于每一个并行的线程块（无论它在算第几行）都需要从 weight 的 第 0 个元素 开始读取
        block_shape=(D_TILE_SIZE,),
        order=(0,)
    )

    # output_block_ptr: 指向 output 张量的输出区域
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,)
    )

    # -------------------------- 2. 初始化输出缓冲区 --------------------------
    # 为当前线程块分配寄存器，存储计算结果，初始化为0
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    # -------------------------- 3. 循环计算（按维度 D 分块） --------------------------
    # tl.cdiv: 向上取整除法，确保处理完所有维度（即使D不能被D_TILE_SIZE整除）
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 1. 加载数据到 SRAM，自动处理边界检查和填充
        # boundary_check: 检查是否越界
        # padding_option="zero": 越界部分填充0
        row = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),  # 检查0和1两个维度
            padding_option="zero"
        )  # 加载形状: [ROWS_TILE_SIZE, D_TILE_SIZE]

        weight = tl.load(
            weight_block_ptr,
            boundary_check=(0,),    # 仅检查列维度
            padding_option="zero"
        )  # 加载形状: [D_TILE_SIZE]

        # 2. 核心计算：加权求和
        # weight[None, :]: 升维为 [1, D_TILE_SIZE]，与 row 进行广播相乘
        # tl.sum(... axis=1): 沿特征维度求和，得到 [ROWS_TILE_SIZE]
        output += tl.sum(row * weight[None, :], axis=1)

        # 3. 移动指针：处理下一个分块（列方向移动 D_TILE_SIZE）
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    # -------------------------- 4. 写回结果 --------------------------
    # 将寄存器中的结果写回 GPU 全局内存
    tl.store(
        output_block_ptr,
        output,
        boundary_check=(0,)  # 处理最后一个块可能的行越界
    )


@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,  # 输入：前向计算的张量 x 和权重 weight
    grad_output_ptr,     # 输入：上游梯度（grad_output）
    # 输出：对 x 的梯度 grad_x，对 weight 的部分梯度 partial_grad_weight
    # 对于某一个元素x(i,j)，它只参与了第 i 行的求和, 所以每个线程块写的grad_x互不重叠
    grad_x_ptr,
    # partial_grad_weight_ptr是对weight求导的结果。权重W(i,j) 参与所有行的计算，不同线程块都想往grad_W里写东西
    # 为了避免冲突，用partial_grad_weight_ptr来保存各个线程自己的梯度
    partial_grad_weight_ptr,

    stride_xr, stride_xd,  # 张量 x 的行和列维度步长
    stride_wd,             # 张量 weight 的维度步长
    stride_gr,             # 张量 grad_output 的维度步长
    stride_gxr, stride_gxd,  # 张量 grad_x 的行和列维度步长
    stride_gwb, stride_gwd,  # 张量 partial_grad_weight 的行和列维度步长
    NUM_ROWS, D,  # 输入维度：行数 NUM_ROWS，特征维度 D
    ROWS_TILE_SIZE: tl.constexpr,  # 编译时常量：行 tile 大小
    D_TILE_SIZE: tl.constexpr      # 编译时常量：特征 tile 大小
):
    # 获取当前程序块（thread block）的索引和总块数
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    # 1. 创建 grad_output 的块指针
    # 形状：(NUM_ROWS,)，步长：stride_gr 形状和前向输出是一样的，所以是block_shape=(ROWS_TILE_SIZE,),
    # 偏移：从当前行 tile 的起始位置开始
    # 块形状：(ROWS_TILE_SIZE,)，顺序：按行加载
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,), strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # 2. 创建输入张量 x 的块指针
    # 形状：(NUM_ROWS, D)，步长：(stride_xr, stride_xd)
    # 偏移：从当前行 tile 的起始位置开始，列从 0 开始
    # 块形状：(ROWS_TILE_SIZE, D_TILE_SIZE)，顺序：先列后行（优化内存访问）
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D), strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # 3. 创建权重张量 weight 的块指针
    # 形状：(D,)，步长：stride_wd
    # 偏移：从 0 开始
    # 块形状：(D_TILE_SIZE,)，顺序：按行加载
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), strides=(stride_wd,),
        offsets=(0,), block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # 4. 创建 grad_x 的块指针
    # 形状：(NUM_ROWS, D)，步长：(stride_gxr, stride_gxd)
    # 偏移：从当前行 tile 的起始位置开始，列从 0 开始
    # 块形状：(ROWS_TILE_SIZE, D_TILE_SIZE)，顺序：先列后行
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D), strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # 5. 创建 partial_grad_weight 的块指针
    # 形状：(n_row_tiles, D)，步长：(stride_gwb, stride_gwd)
    # 偏移：从当前行 tile 的行索引开始，列从 0 开始
    # 块形状：(1, D_TILE_SIZE)，顺序：先列后行
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D,), strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    # 按 D 维度 tile 循环，处理所有特征维度
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 加载当前 grad_output tile：形状 (ROWS_TILE_SIZE,)
        grad_output = tl.load(grad_output_block_ptr,
                              boundary_check=(0,), padding_option="zero")
        # 加载当前 weight tile：形状 (D_TILE_SIZE,)
        weight = tl.load(weight_block_ptr, boundary_check=(
            0,), padding_option="zero")
        # 加载当前 x tile：形状 (ROWS_TILE_SIZE, D_TILE_SIZE)
        row = tl.load(x_block_ptr, boundary_check=(
            0, 1), padding_option="zero")

        # 计算 grad_x：外积 grad_x = grad_output[:, None] * weight[None, :]
        # 计算外积：形状 (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_x_row = grad_output[:, None] * weight[None, :]
        # 存储到 grad_x 的对应位置
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # 计算 partial_grad_weight：对 x * grad_output 按行求和
        # 逐元素相乘后，按行（axis=0）求和，得到形状 (1, D_TILE_SIZE)
        grad_weight_row = tl.sum(
            row * grad_output[:, None], axis=0, keep_dims=True)
        # axis=0 沿着行压缩，意思是把行12345压缩（加和）
        # 存储到 partial_grad_weight 的对应位置
        tl.store(partial_grad_weight_block_ptr,
                 grad_weight_row, boundary_check=(1,))

        # 移动所有指针到下一个 D 维度 tile
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance(
            (0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


# 功能：将 Triton 核函数封装为 PyTorch 可求导的 Function


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        """
        前向传播接口
        Args:
            ctx: 上下文对象，用于保存反向传播需要的变量
            x: 输入张量，形状为 [*, D]（任意维度，最后一维为特征维）
            weight: 权重张量，形状为 [D]
        Returns:
            y: 加权求和结果，形状为 [*]（去掉了最后一维）
        """
        # 检查输入合法性
        assert len(weight.shape) == 1 and weight.shape[0] == x.shape[-1], \
            "Dimension mismatch: weight.shape[0] must equal x.shape[-1]"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        # -------------------------- 1. 维度处理与重塑 --------------------------
        # 保存原始形状，用于输出时恢复
        input_shape = x.shape
        # 将 x 展平为 2D 张量 [N, D]，方便 Triton 处理（N为展平后的总行数）
        # rearrange: " ... d -> (...) d"  将除最后一维外的所有维度展平为一维
        x_2d = rearrange(x, "... d -> (...) d")
        N, D = x_2d.shape  # N: 展平后的行数, D: 特征维度

        # -------------------------- 2. 配置 Triton 编译参数 --------------------------
        # D_TILE_SIZE: 特征维度的分块大小，取2的幂次（H100架构友好）
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
        # ROWS_TILE_SIZE: 每线程块处理的行数，这里固定为16
        ctx.ROWS_TILE_SIZE = 16
        # 保存反向传播需要的变量
        ctx.save_for_backward(x_2d, weight)
        ctx.input_shape = input_shape  # 保存原始形状

        # -------------------------- 3. 初始化输出张量 --------------------------
        y = torch.empty(N, device=x.device, dtype=x.dtype)

        # -------------------------- 4. 启动 Triton 核函数 --------------------------
        # 计算需要的线程块数量：ceil(N / ROWS_TILE_SIZE)
        def grid(META): return (triton.cdiv(N, META['ROWS_TILE_SIZE']),)

        # 调用 Triton 核函数
        # 我们会定义一个所谓的“启动网格”的线程块。然后，我们可以在内核中通过 tl.program_id(0) 访问线程块索引。
        weighted_sum_fwd[grid](
            x_2d, weight,                # 输入输出张量
            y,
            # 步幅设置：x_2d的行步幅是D（每行有D个元素），列步幅是1
            x_2d.stride(0), x_2d.stride(1),
            weight.stride(0),            # weight的步幅
            y.stride(0),                 # y的步幅
            ROWS=N, D=D,                 # 实际维度
            # 编译时常量
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE
        )

        # -------------------------- 5. 恢复形状并返回 --------------------------
        # 将 2D 的输出 y 恢复为输入的原始形状（去掉最后一维）
        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        # 从上下文中恢复前向保存的张量和 tile 大小
        x, weight = ctx.saved_tensors
        # 行 tile 和特征 tile 大小可不同
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape  # 获取输入张量 x 的形状

        # 策略：每个线程块先写入部分梯度缓冲区，再对缓冲区进行归约得到最终梯度
        # 初始化部分梯度缓冲区 partial_grad_weight：形状 (n_row_tiles, D)
        partial_grad_weight = torch.empty(
            (triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        # 初始化 grad_x：与 x 形状相同
        grad_x = torch.empty_like(x)

        # 调用 Triton 核函数执行反向传播
        # 网格大小：按行 tile 划分
        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )

        # 对 partial_grad_weight 按行归约，得到最终的 grad_weight
        grad_weight = partial_grad_weight.sum(axis=0)
        # 返回对 x 和 weight 的梯度
        return grad_x, grad_weight


f_weightedsum = WeightedSumFunc.apply


def weighted_sum(x, weight):
    # Here, assume that x has n-dim shape [..., D], and weight has 1D shape [D]
    # return (weight * x).sum(axis=-1)
    return f_weightedsum(x, weight)


if __name__ == "__main__":
    batch_size = 8
    seq_len = 1024
    d_model = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 准备数据
    x = torch.randn(batch_size, seq_len, d_model,
                    device=device, requires_grad=True)

    # 【核心修改点】：你的 Kernel 逻辑是 y = sum(x * weight, dim=-1)
    # 这里的 weight 必须是 1D 向量 [16]，才能匹配你在 Forward 里的 weight_block_ptr 逻辑
    W = torch.randn(d_model, device=device, requires_grad=True) * 0.01

    # 2. 调用函数
    # 现在 x 是 [8, 1024, 16], W 是 [16]
    # 输出 res 应该是 [8, 1024]
    res = weighted_sum(x, W)

    print(res)
    print(f"输出形状: {res.shape}")  # 预期: torch.Size([8, 1024])
