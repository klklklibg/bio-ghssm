"""
Bio-GHSSM SSMCell — 向量化 SSM，性能优化版。
x → SSM scan → output
双缓冲: W_slow 冻结锚点，W_fast 可训练，forward 用 gate 加权平均。
向量化 scan：使用 cumsum 替代 for 循环，大幅提速。
"""
import torch
import torch.nn as nn


class BioSSMCell(nn.Module):
    """
    向量化 SSM Cell with dual-buffer weights.

    架构:
      x → (x_proj)  (d_model → ssm_dim)
      SSM scan (vectorized): h[t] = A * h[t-1] + x_proj[t]
      y = h @ W_s2y

    双缓冲:
      - W_slow: 冻结锚点，Sleep 期通过 replay 梯度更新
      - W_fast: 可训练，Wake 期学习当前任务
      - Forward 时: W = (1-g)*W_slow + g*W_fast
    """

    def __init__(self, d_model: int, ssm_dim: int):
        super().__init__()
        self.d_model = d_model
        self.ssm_dim = ssm_dim

        # 双缓冲: x → ssm_dim
        self.W_slow_x2s = nn.Parameter(torch.randn(d_model, ssm_dim) * 0.02)
        self.W_fast_x2s = nn.Parameter(torch.randn(d_model, ssm_dim) * 0.02)

        # 双缓冲: SSM 输出 → d_model
        self.W_slow_s2y = nn.Parameter(torch.randn(ssm_dim, d_model) * 0.02)
        self.W_fast_s2y = nn.Parameter(torch.randn(ssm_dim, d_model) * 0.02)

        # SSM 状态矩阵 A: 对角线初始化
        A_init = torch.eye(ssm_dim, ssm_dim) * 0.8
        self.A = nn.Parameter(A_init)

        # 整合 gate
        self.W_gate = nn.Parameter(torch.tensor(0.5))

        # 冻结 W_slow
        self.W_slow_x2s.requires_grad = False
        self.W_slow_s2y.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq, d_model)
        返回: (batch, seq, d_model)
        """
        batch, seq_len, _ = x.shape

        # 整合权重
        g = torch.sigmoid(self.W_gate).clamp(0, 1)
        W_x2s = (1 - g) * self.W_slow_x2s + g * self.W_fast_x2s
        W_s2y = (1 - g) * self.W_slow_s2y + g * self.W_fast_s2y

        # SSM 投影
        x_proj = x @ W_x2s  # (batch, seq, ssm_dim)

        # 状态矩阵 A（提取对角线为向量）
        A = torch.diagonal(torch.exp(torch.abs(self.A)) * 0.8).contiguous()  # (ssm_dim,)

        # t_idx: [seq-1, seq-2, ..., 1, 0]
        # A_powers[t, d] = A[d] ** t_idx[t]
        t_idx = torch.arange(seq_len - 1, -1, -1, device=x.device, dtype=torch.float32)  # (seq,)
        A_powers = torch.pow(A.unsqueeze(0), t_idx.unsqueeze(-1))  # (seq, ssm_dim)

        # 向量化 scan: h[t] = sum_{i=0}^{t} A^{t-i} * x_proj[i]
        x_rev = x_proj.flip(1)  # (batch, seq, ssm_dim)
        weighted = x_rev * A_powers.unsqueeze(0)  # (batch, seq, ssm_dim)
        h = weighted.cumsum(dim=1).flip(1)  # (batch, seq, ssm_dim)

        # 输出投影
        y = h @ W_s2y  # (batch, seq, d_model)

        return y

    def drift(self):
        return (
            (self.W_fast_x2s - self.W_slow_x2s).pow(2).mean().item() +
            (self.W_fast_s2y - self.W_slow_s2y).pow(2).mean().item()
        )

    def init_slow_from_fast(self):
        with torch.no_grad():
            self.W_slow_x2s.copy_(self.W_fast_x2s)
            self.W_slow_s2y.copy_(self.W_fast_s2y)
