"""
Bio-GHSSM 独立模型训练器 — Wake-Sleep + 外部记忆持续学习协议.
修复: train_task 支持 collate 返回 (x,y) tuple 或单个 tensor
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random, json, time, math
from pathlib import Path
from .model import BioGHSSMModel
from .external_memory import ExternalMemory


class RandomBuffer:
    """简单的随机抽样 episodic buffer."""
    
    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, tokens: torch.Tensor):
        for i in range(tokens.size(0)):
            self.buffer.append(tokens[i].cpu().clone())
        while len(self.buffer) > self.capacity:
            remove_idx = random.randint(0, len(self.buffer) - 1)
            self.buffer.pop(remove_idx)
    
    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            return None
        indices = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
        return torch.stack([self.buffer[i] for i in indices])


class TokenDataset(Dataset):
    def __init__(self, tokens_list):
        self.tokens = tokens_list
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        return self.tokens[idx]


def collate_fn(batch):
    return torch.stack(batch)


class BioGHSSMTrainer:
    """
    Wake-Sleep 训练器，支持可选的外部记忆。
    
    Wake Phase: W_fast 学习当前任务，存储样本到 buffer
    Sleep Phase: W_slow 通过 replay 梯度更新
    External Memory: 每次任务训练后存储 W_slow 快照
                     评估时用 blend 公式混合历史快照回 W_slow
                     公式: W_new = (1 - n*α)*W_current + α*(snap1+...+snapN)
    
    Args:
        use_external_memory: 是否启用外部记忆（默认 False，保持向后兼容）
        memory_alpha: blend 系数（推荐 0.20~0.40，默认 0.20）
        memory_capacity: 最大存储快照数（默认 10）
    """
    
    def __init__(
        self,
        model: BioGHSSMModel,
        buffer_size: int = 200,
        wake_lr: float = 1e-3,
        sleep_lr: float = 1e-4,
        sleep_steps: int = 60,
        device: str = 'cuda',
        seed: int = 42,
        # ── 外部记忆配置 ──
        use_external_memory: bool = False,
        memory_alpha: float = 0.20,
        memory_capacity: int = 10,
    ):
        self.model = model
        self.device = device
        self.seed = seed
        self.sleep_steps = sleep_steps
        self.sleep_lr = sleep_lr
        self.wake_lr = wake_lr
        
        # 外部记忆
        self.use_external_memory = use_external_memory
        self.memory_alpha = memory_alpha
        self.memory = None
        self._slow_names = None  # 延迟初始化
        
        # 统一设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.buffer = RandomBuffer(capacity=buffer_size)
        self.model.init_slow_from_fast()
        
        self.history = []
        self._init_ppls = {}
        self._memory_init_done = False  # 等 init_slow_from_fast 后才初始化
    
    def _ensure_memory_init(self):
        """延迟初始化外部记忆（在 init_slow_from_fast 后）"""
        if self._memory_init_done:
            return
        if self._slow_names is None:
            self._slow_names = [n for n, p in self.model.named_parameters() if 'W_slow' in n]
        if self.use_external_memory:
            self.memory = ExternalMemory(
                slow_param_names=self._slow_names,
                capacity=10,
                device=str(self.device),
            )
            self.memory.set_shapes(self.model.named_parameters())
        self._memory_init_done = True
    
    def _slow_params(self):
        return [p for n, p in self.model.named_parameters() if 'W_slow' in n]
    
    def _fast_params(self):
        return [p for n, p in self.model.named_parameters() if 'W_fast' in n]
    
    def _other_params(self):
        names = {'W_gate', 'A', 'W_x2i', 'W_x2f', 'W_x2g', 'W_x2o', 'b_'}
        return [p for n, p in self.model.named_parameters()
                if any(ns in n for ns in names) and 'W_slow' not in n and 'W_fast' not in n]
    
    def _unpack_batch(self, batch):
        """支持两种格式: 单tensor(B,T) 或 tuple(x,y)"""
        if isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[1]
        else:
            x = batch
            y = batch  # 模型内部会做 shift
        return x.to(self.device), y.to(self.device)
    
    def _eval_ppl(self, dataloader):
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in dataloader:
                x, y = self._unpack_batch(batch)
                loss = self.model(x, y)
                total += loss.item() * x.size(0)
                count += x.size(0)
        return math.exp(total / max(count, 1))
    
    def train_task(self, task_id: int, dataloader: DataLoader, epochs: int = 3,
                   save_snapshot: bool = True):
        """
        完整训练单个任务: wake → sleep → (可选)写外部记忆快照。
        
        save_snapshot=True 时，任务训练结束后自动将 W_slow 写入外部记忆。
        """
        self._ensure_memory_init()
        
        # ── Wake Phase ──
        self.model.freeze_all()
        for p in self._fast_params():
            p.requires_grad = True
        for p in self._other_params():
            p.requires_grad = True
        
        opt = torch.optim.AdamW(
            list(self._fast_params()) + list(self._other_params()),
            lr=self.wake_lr,
            foreach=True,
        )
        
        t0 = time.time()
        init_loss = None
        final_loss = None
        
        for epoch in range(epochs):
            self.model.train()
            for batch in dataloader:
                x, y = self._unpack_batch(batch)
                opt.zero_grad()
                loss = self.model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                self.buffer.add(x.cpu())
            
            self.model.eval()
            total_loss = 0
            count = 0
            with torch.no_grad():
                for batch in dataloader:
                    x, y = self._unpack_batch(batch)
                    loss = self.model(x, y)
                    total_loss += loss.item() * x.size(0)
                    count += x.size(0)
            avg_loss = total_loss / max(count, 1)
            if init_loss is None:
                init_loss = avg_loss
            final_loss = avg_loss
        
        wake_loss = final_loss
        
        # ── Sleep Phase ──
        self.model.freeze_all()
        for p in self._slow_params():
            p.requires_grad = True
        
        sleep_opt = torch.optim.AdamW(self._slow_params(), lr=self.sleep_lr, foreach=True)
        
        n = self.sleep_steps
        step_size = max(1, len(self.buffer) // n)
        
        for step in range(n):
            batch = self.buffer.sample(min(step_size, 32))
            if batch is None:
                break
            x = batch.to(self.device)
            y = batch.to(self.device)
            sleep_opt.zero_grad()
            loss = self.model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            sleep_opt.step()
        
        # ── 写外部记忆快照 ──
        if save_snapshot and self.use_external_memory and self.memory is not None:
            slow_state = {n: p.detach().clone() for n, p in self.model.named_parameters()
                          if 'W_slow' in n}
            self.memory.write(task_id, slow_state)
        
        drift = self.model.drift()
        elapsed = time.time() - t0
        
        # ── 最终 loss ──
        self.model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                x, y = self._unpack_batch(batch)
                loss = self.model(x, y)
                total_loss += loss.item() * x.size(0)
                count += x.size(0)
        final_loss = total_loss / max(count, 1)
        
        forgetting = wake_loss - final_loss
        
        record = {
            'task': f'task_{task_id}',
            'init_loss': init_loss,
            'wake_loss': wake_loss,
            'final_loss': final_loss,
            'forgetting': forgetting,
            'drift': drift,
            'time_s': elapsed,
            'memory_snaps': len(self.memory) if self.memory else 0,
        }
        self.history.append(record)
        
        print(f'  [task_{task_id}] wake={wake_loss:.4f} sleep={final_loss:.4f} '
              f'forgetting={forgetting:+.4f} drift={drift:.6f} [{elapsed:.1f}s]')
        
        self._init_ppls[task_id] = init_loss
        return record
    
    def evaluate_all(self, dataloaders: dict, use_memory: bool = False,
                     alpha: float = None, selective: bool = False):
        """
        评估所有任务。
        
        Args:
            dataloaders: {task_name: DataLoader} 字典
            use_memory: 是否在评估时应用外部记忆 blend
            alpha: blend 系数（覆盖 self.memory_alpha，用于 sweep）
            selective: 是否排除当前任务的快照（默认 False，blend 所有历史快照）
        """
        self._ensure_memory_init()
        
        if alpha is None:
            alpha = self.memory_alpha
        
        results = {}
        orig_state = {n: self.model.state_dict()[n].clone()
                      for n in self._slow_names} if self._slow_names else {}
        
        for eval_tid, (task_name, dl) in enumerate(dataloaders.items(), start=1):
            if use_memory and self.memory is not None and len(self.memory) > 0:
                exclude = eval_tid if selective else None
                self.memory.blend(self.model.state_dict(), alpha=alpha,
                                  selective=selective, exclude_task=exclude)
            
            ppl = self._eval_ppl(dl)
            results[task_name] = ppl
            
            # 恢复原始 W_slow
            if use_memory and self.memory is not None and len(self.memory) > 0:
                for n in self._slow_names:
                    self.model.state_dict()[n].copy_(orig_state[n])
        
        return results
    
    def evaluate_task(self, task_name: str, dataloader: DataLoader,
                     use_memory: bool = False, alpha: float = None):
        """评估单个任务"""
        self._ensure_memory_init()
        
        if alpha is None:
            alpha = self.memory_alpha
        
        orig_state = {n: self.model.state_dict()[n].clone()
                      for n in self._slow_names} if self._slow_names else {}
        
        if use_memory and self.memory is not None and len(self.memory) > 0:
            task_id = int(task_name.split('_')[-1]) if '_' in task_name else None
            self.memory.blend(self.model.state_dict(), alpha=alpha,
                              selective=True, exclude_task=task_id)
        
        ppl = self._eval_ppl(dataloader)
        
        for n in self._slow_names:
            self.model.state_dict()[n].copy_(orig_state[n])
        
        return ppl
    
    def set_memory_alpha(self, alpha: float):
        """动态修改 memory_alpha（用于 alpha sweep）"""
        self.memory_alpha = alpha
    
    def enable_memory(self, enabled: bool = True):
        """开关外部记忆"""
        self.use_external_memory = enabled
        if enabled and not self._memory_init_done:
            self._ensure_memory_init()
    
    def get_memory_stats(self):
        """返回记忆统计"""
        if self.memory is None:
            return {'enabled': False, 'n_snapshots': 0}
        return {
            'enabled': True,
            'n_snapshots': len(self.memory),
            'alpha': self.memory_alpha,
            'task_ids': list(self.memory.slots.keys()),
        }
    
    def save(self, output_dir: str, name: str = 'bio_ghssm_ind'):
        import tempfile
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ckpt_path = output_dir / f'{name}_checkpoint.pt'
        tmp = output_dir / f'{name}_checkpoint.pt.tmp'
        torch.save({
            'model_state': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'history': self.history,
            'config': dict(
                vocab_size=self.model.vocab_size,
                d_model=self.model.d_model,
                ssm_dim=self.model.ssm_dim,
                n_layers=self.model.n_layers,
            )
        }, tmp)
        tmp.rename(ckpt_path)
        
        self.model.merge_weights()
        merged_path = output_dir / f'{name}_merged.pt'
        tmp2 = output_dir / f'{name}_merged.pt.tmp'
        torch.save({k: v.cpu() for k, v in self.model.state_dict().items()}, tmp2)
        tmp2.rename(merged_path)
        
        with open(output_dir / f'{name}_summary.json', 'w', encoding='utf-8') as f:
            json.dump({'history': self.history, 'config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'ssm_dim': self.model.ssm_dim,
                'n_layers': self.model.n_layers,
            }}, f, indent=2)
        
        print(f'  Checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.2f} MB)')
        print(f'  Merged:     {merged_path} ({merged_path.stat().st_size / 1e6:.2f} MB)')