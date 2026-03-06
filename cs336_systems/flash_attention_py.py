import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from jaxtyping import Bool, Float, Int
from einops import einsum
from math import exp


class PythonFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FlashAttention-2 正向传播的纯 PyTorch 实现 (Tiling 逻辑)
        """
        # 初始化
        O = torch.zeros_like(Q)
        # m 记录每行的最大值，初始化为负无穷
        m = torch.full((*Q.shape[:-1], 1), float("-inf"),
                       device=Q.device, dtype=Q.dtype)
        # l 记录每行的 exp 累加值，初始化为 0
        l = torch.zeros((*Q.shape[:-1], 1), device=Q.device, dtype=Q.dtype)
        # m = torch.full((Q.shape[0], Q.shape[1], 1), float("-inf"), device=Q.device, dtype=Q.dtype)
        # l = torch.zeros((Q.shape[0], Q.shape[1], 1), device=Q.device, dtype=Q.dtype)

        # 确定 Tile 大小 (讲义要求至少 16x16)
        Br = 16
        Bc = 16
        Batch, Seq, Dim = Q.shape

        # 在seq上循环而不是Dim上！ 至少得是二维吧！
        # 修正：向上取整确保覆盖所有 Token
        rows = (Seq + Br - 1) // Br
        columns = (Seq + Bc - 1) // Bc

        # 缩放因子 (1/sqrt(d))
        sm_scale = Dim ** -0.5

        for j in range(columns):
            # 加载 K, V 的列块 [Batch, Head, Bc, Dim]，注意取min防止越界
            j_start, j_end = j * Bc, min((j + 1) * Bc, Seq)
            kj = K[:, j_start:j_end, :]
            vj = V[:, j_start:j_end, :]

            for i in range(rows):
                cur_row_slice = slice(
                    i * Br, min((i + 1) * Br, Seq))  # 注意取min防止越界
                # 加载 Q 的行块 [Batch, Head, Br, Dim]
                qi = Q[:, cur_row_slice, :]

                # 计算得分 S_ij = Q_i @ K_j^T  [Batch, Head, Br, Bc]。注意缩放sm_scale
                # 注意最后两维的转置, 只转置最后两维!! 所以不用一开始上来就K.T
                score = torch.matmul(qi, kj.transpose(-1, -2)) * sm_scale

                # --- online Softmax ---
                # 计算当前块的最大值
                m_ij = torch.max(score, dim=-1, keepdim=True).values
                # 更新全局最大值
                m_new = torch.max(m[:, cur_row_slice, :], m_ij)

                # 过去的值更新exp的话需要
                exp_update = m[:, cur_row_slice, :] - m_new
                # 现在的值计算exp的话需要
                exp_cur = score - m_new

                l_cur = torch.sum(torch.exp(exp_cur), dim=-1, keepdim=True)
                l_old = l[:, cur_row_slice, :]
                l_update = l_old * torch.exp(exp_update)

                # 计算当前块未归一化的得分
                cur_o = torch.matmul(torch.exp(exp_cur), vj)

                # 更新 O (Output), 因为O里也有max
                # fmt:off
                updated_old_o =  O[:, cur_row_slice,: ]*torch.exp(exp_update)
                # fmt:on

                # alpha = torch.exp(exp_update)
                # 我们把旧的 O 缩放对齐到 m_new，然后加上当前块的贡献
                O[:, cur_row_slice, :] = updated_old_o + cur_o

                # 3. 更新全局统计量 l 和 m
                # l_update 你已经算过了：l_old * exp(m_old - m_new)
                l[:, cur_row_slice, :] = l_update + l_cur
                m[:, cur_row_slice, :] = m_new

        # --- 整个 K,V 外循环结束后 ---
        # 别忘了最后要做一次分母归一化
        O = O / l
        # 将 l 的 log 值存入 L 供反向传播使用 (logsumexp)
        # +m是为了防溢出，因为score往往很大。这样操作了之后L就是我们计算梯度用的那个L了
        L = m + torch.log(l)

        L_saved = L.squeeze(-1)

        # -----------------------------------------------------------

        # 3. 保存上下文用于反向传播
        ctx.save_for_backward(Q, K, V, O, L_saved)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_output):
        # 即使逻辑没写，也要返回全零张量，否则 grad 属性是 None，测试直接崩
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        # grad_output 是 dO [Batch, Seq, Dim]
        dO = grad_output
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # 对应 forward 的输入: Q, K, V, is_causal
        return dQ, dK, dV, None


def flash_attention_pytorch(Q, K, V, is_causal=False):
    # 修改：适配只返回一个 O 的 forward
    O = PythonFlashAttentionFunction.apply(Q, K, V, is_causal)
    return O
