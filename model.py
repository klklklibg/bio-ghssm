"""
Bio-GHSSM 独立模型 — 堆叠 SSM Block + Embedding + Head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cell import BioSSMCell


class BioSSMBlock(nn.Module):
    """单层 SSM Block: BioSSMCell + LayerNorm + 残差连接.
    
    每 block 内部维护自己的 h（ssm_dim），每个序列位置独立扫描。
    """
    
    def __init__(self, d_model: int, ssm_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ssm = BioSSMCell(d_model, ssm_dim)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """x: (batch, seq, d_model)"""
        residual = x
        out = self.ssm(x)   # 每 block 内部 scan
        out = self.dropout(out)
        out = self.ln(out + residual)
        return out


class BioGHSSMModel(nn.Module):
    """
    独立 Bio-GHSSM 模型 — SSM-first 架构.
    
    架构: Embedding → BioSSMBlock×N → LayerNorm → Head
    
    双缓冲机制:
      - 每层 SSM Block 有 W_slow (冻结) + W_fast (可训练)
      - Wake 期: W_slow 冻结，W_fast 学习当前任务
      - Sleep 期: W_slow 可训练，W_fast 冻结，回放历史梯度
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        ssm_dim: int = 32,
        n_layers: int = 4,
        dropout: float = 0.1,
        padding_idx: int = 0,
        ignore_index: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.ssm_dim = ssm_dim
        self.n_layers = n_layers
        self.ignore_index = ignore_index
        
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        
        self.blocks = nn.ModuleList([
            BioSSMBlock(d_model, ssm_dim, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重绑定
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def num_total_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        """
        input_ids: (batch, seq_len)
        targets: (batch, seq_len) — 如果传入则计算 loss
        """
        x = self.embed(input_ids)  # (batch, seq, d_model)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln(x)
        logits = self.head(x)  # (batch, seq, vocab)
        
        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.ignore_index,
            )
            return loss
        
        return logits
    
    def slow_param_groups(self):
        """返回所有 W_slow 参数组（用于 Sleep 期训练）。"""
        return [
            {'params': [p], 'lr': 1e-4}
            for p in self.parameters()
            if not p.requires_grad
        ]
    
    def fast_param_groups(self):
        """返回所有 W_fast 参数组（用于 Wake 期训练）。"""
        return [
            {'params': [p], 'lr': 1e-3}
            for p in self.parameters()
            if p.requires_grad
        ]
    
    def all_param_groups(self):
        """返回所有参数组（lr 相同）。"""
        return [{'params': list(self.parameters())}]
    
    def init_slow_from_fast(self):
        for block in self.blocks:
            block.ssm.init_slow_from_fast()
    
    def drift(self):
        return sum(b.ssm.drift() for b in self.blocks) / self.n_layers
    
    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def unfreeze_slow(self):
        for p in self.parameters():
            if not p.requires_grad:
                p.requires_grad = True
    
    def unfreeze_fast(self):
        for p in self.parameters():
            p.requires_grad = True
    
    def freeze_slow(self):
        for p in self.parameters():
            if not p.requires_grad:
                pass  # already frozen
            else:
                # 检查是否是 slow 参数
                name = p.name if hasattr(p, 'name') else ''
                if 'W_slow' in name or ('slow' in name.lower() and p.grad is None):
                    p.requires_grad = False
    
    @torch.no_grad()
    def merge_weights(self):
        """合并 W_slow + W_fast 到 W_fast（推理导出）。"""
        for block in self.blocks:
            cell = block.ssm
            g = torch.sigmoid(cell.W_gate).clamp(0, 1).item()
            # x2s
            W_x2s = (1 - g) * cell.W_slow_x2s + g * cell.W_fast_x2s
            cell.W_fast_x2s.copy_(W_x2s)
            # s2y
            W_s2y = (1 - g) * cell.W_slow_s2y + g * cell.W_fast_s2y
            cell.W_fast_s2y.copy_(W_s2y)
            # 冻结 slow
            cell.W_slow_x2s.requires_grad = False
            cell.W_slow_s2y.requires_grad = False
