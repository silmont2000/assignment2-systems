# import triton
# import triton.language as tl
# import torch
# from einops import rearrange  # 代码中使用了rearrange，需导入该库

# # -------------------------- 第一步：定义 Triton 核函数 --------------------------
# # 功能：在 GPU 上并行计算加权求和，y = sum(x * weight, dim=-1)


# @triton.jit
# def weighted_sum_fwd(
#     x_ptr, weight_ptr,          # 输入张量的指针 (x: [N, D], weight: [D])
#     output_ptr,                 # 输出张量的指针 (output: [N])
#     x_stride_row, x_stride_dim,  # x张量的行步幅和维度步幅
#     weight_stride_dim,          # weight张量的步幅（通常为1）
#     output_stride_row,          # output张量的步幅（通常为1）
#     ROWS, D,                    # 张量的实际维度：ROWS=N(行), D=特征维度
#     # 编译时常量，决定每个线程块处理的数据大小（Tile）
#     ROWS_TILE_SIZE: tl.constexpr,
#     D_TILE_SIZE: tl.constexpr,
# ):
#     """
#     Triton 前向传播核：计算每一行的加权和
#     """
#     # 获取当前线程块的 ID，用于定位该块需要处理的行范围
#     row_tile_idx = tl.program_id(0)

#     # -------------------------- 1. 构建块指针 (Block Pointers) --------------------------
#     # 块指针是 Triton 特有的内存管理方式，用于高效地加载/存储连续或跨步的内存块
#     # x_block_ptr: 指向 x 张量中当前线程块需要处理的区域
#     x_block_ptr = tl.make_block_ptr(
#         base=x_ptr,
#         shape=(ROWS, D),                  # 张量的【整体】形状
#         strides=(x_stride_row, x_stride_dim),  # 内存布局的步幅
#         # 起始偏移：行方向偏移为 块ID * 块大小，列方向从0开始
#         offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
#         block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),  # 每次加载的块大小
#         order=(1, 0)                      # 内存访问顺序：先列后行（适配GPU内存合并）
#     )

#     # weight_block_ptr: 指向 weight 张量
#     weight_block_ptr = tl.make_block_ptr(
#         base=weight_ptr,
#         shape=(D,),
#         strides=(weight_stride_dim,),
#         offsets=(0,),
#         block_shape=(D_TILE_SIZE,),
#         order=(0,)
#     )

#     # output_block_ptr: 指向 output 张量的输出区域
#     output_block_ptr = tl.make_block_ptr(
#         base=output_ptr,
#         shape=(ROWS,),
#         strides=(output_stride_row,),
#         offsets=(row_tile_idx * ROWS_TILE_SIZE,),
#         block_shape=(ROWS_TILE_SIZE,),
#         order=(0,)
#     )

#     # -------------------------- 2. 初始化输出缓冲区 --------------------------
#     # 为当前线程块分配寄存器，存储计算结果，初始化为0
#     output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

#     # -------------------------- 3. 循环计算（按维度 D 分块） --------------------------
#     # tl.cdiv: 向上取整除法，确保处理完所有维度（即使D不能被D_TILE_SIZE整除）
#     for i in range(tl.cdiv(D, D_TILE_SIZE)):
#         # 1. 加载数据到 SRAM，自动处理边界检查和填充
#         # boundary_check: 检查是否越界
#         # padding_option="zero": 越界部分填充0
#         row = tl.load(
#             x_block_ptr,
#             boundary_check=(0, 1),  # 检查行和列两个维度
#             padding_option="zero"
#         )  # 加载形状: [ROWS_TILE_SIZE, D_TILE_SIZE]

#         weight = tl.load(
#             weight_block_ptr,
#             boundary_check=(0,),    # 仅检查列维度
#             padding_option="zero"
#         )  # 加载形状: [D_TILE_SIZE]

#         # 2. 核心计算：加权求和
#         # weight[None, :]: 升维为 [1, D_TILE_SIZE]，与 row 进行广播相乘
#         # tl.sum(... axis=1): 沿特征维度求和，得到 [ROWS_TILE_SIZE]
#         output += tl.sum(row * weight[None, :], axis=1)

#         # 3. 移动指针：处理下一个分块（列方向移动 D_TILE_SIZE）
#         x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
#         weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

#     # -------------------------- 4. 写回结果 --------------------------
#     # 将寄存器中的结果写回 GPU 全局内存
#     tl.store(
#         output_block_ptr,
#         output,
#         boundary_check=(0,)  # 处理最后一个块可能的行越界
#     )

# # -------------------------- 第二步：PyTorch 自定义自动求导封装 --------------------------
# # 功能：将 Triton 核函数封装为 PyTorch 可求导的 Function


# class WeightedSumFunc(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, weight):
#         """
#         前向传播接口
#         Args:
#             ctx: 上下文对象，用于保存反向传播需要的变量
#             x: 输入张量，形状为 [*, D]（任意维度，最后一维为特征维）
#             weight: 权重张量，形状为 [D]
#         Returns:
#             y: 加权求和结果，形状为 [*]（去掉了最后一维）
#         """
#         # 检查输入合法性
#         assert len(weight.shape) == 1 and weight.shape[0] == x.shape[-1], \
#             "Dimension mismatch: weight.shape[0] must equal x.shape[-1]"
#         assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
#         assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

#         # -------------------------- 1. 维度处理与重塑 --------------------------
#         # 保存原始形状，用于输出时恢复
#         input_shape = x.shape
#         # 将 x 展平为 2D 张量 [N, D]，方便 Triton 处理（N为展平后的总行数）
#         # rearrange: " ... d -> (...) d"  将除最后一维外的所有维度展平为一维
#         x_2d = rearrange(x, "... d -> (...) d")
#         N, D = x_2d.shape  # N: 展平后的行数, D: 特征维度

#         # -------------------------- 2. 配置 Triton 编译参数 --------------------------
#         # D_TILE_SIZE: 特征维度的分块大小，取2的幂次（H100架构友好）
#         ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
#         # ROWS_TILE_SIZE: 每线程块处理的行数，这里固定为16
#         ctx.ROWS_TILE_SIZE = 16
#         # 保存反向传播需要的变量
#         ctx.save_for_backward(x_2d, weight)
#         ctx.input_shape = input_shape  # 保存原始形状

#         # -------------------------- 3. 初始化输出张量 --------------------------
#         y = torch.empty(N, device=x.device, dtype=x.dtype)

#         # -------------------------- 4. 启动 Triton 核函数 --------------------------
#         # 计算需要的线程块数量：ceil(N / ROWS_TILE_SIZE)
#         def grid(META): return (triton.cdiv(N, META['ROWS_TILE_SIZE']),)

#         # 调用 Triton 核函数
#         # 我们会定义一个所谓的“启动网格”的线程块。然后，我们可以在内核中通过 tl.program_id(0) 访问线程块索引。
#         weighted_sum_fwd[grid](
#             x_2d, weight,                # 输入输出张量
#             y,
#             # 步幅设置：x_2d的行步幅是D（每行有D个元素），列步幅是1
#             x_2d.stride(0), x_2d.stride(1),
#             weight.stride(0),            # weight的步幅
#             y.stride(0),                 # y的步幅
#             ROWS=N, D=D,                 # 实际维度
#             # 编译时常量
#             ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
#             D_TILE_SIZE=ctx.D_TILE_SIZE
#         )

#         # -------------------------- 5. 恢复形状并返回 --------------------------
#         # 将 2D 的输出 y 恢复为输入的原始形状（去掉最后一维）
#         return y.view(input_shape[:-1])

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         反向传播接口（图片中未显示完整实现，此处留空待补充）
#         Args:
#             ctx: 上下文对象，包含前向传播保存的变量
#             grad_output: 输出张量的梯度，形状与 forward 输出一致
#         Returns:
#             grad_x: x的梯度，形状与x一致
#             grad_weight: weight的梯度，形状与weight一致
#         """
#         # 从上下文中取出前向传播的输入
#         x, weight = ctx.saved_tensors
#         # TODO: 实现反向传播的 Triton 核调用或使用 PyTorch 计算
#         # 此处需补充梯度计算逻辑
#         grad_x = None
#         grad_weight = None
#         return grad_x, grad_weight


# def weighted_sum(x, weight):
#     # Here, assume that x has n-dim shape [..., D], and weight has 1D shape [D]
#     return (weight * x).sum(axis=-1)