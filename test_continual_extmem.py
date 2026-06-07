"""Bio-GHSSM 外部记忆持续学习验证"""
import torch, sys, time
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, '.')
from independent.model import BioGHSSMModel
from independent.trainer import BioGHSSMTrainer, collate_fn

class RandDataset(Dataset):
    def __init__(self, size, seq_len, vocab):
        self.data = [torch.randint(1, vocab, (seq_len,)) for _ in range(size)]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

model = BioGHSSMModel(vocab_size=200, d_model=128, ssm_dim=32, n_layers=2)
print(f'模型: {model.num_total_params()/1e6:.2f}M 参数')

# 开启外部记忆
trainer = BioGHSSMTrainer(
    model, device='cpu', buffer_size=100,
    wake_lr=1e-3, sleep_lr=1e-4, sleep_steps=30,
    use_external_memory=True,
    memory_alpha=0.25,
    memory_capacity=10,
)
print(f'外部记忆: 开启 (alpha={trainer.memory_alpha})')

# 训练 5 个任务
for tid in range(5):
    ds = RandDataset(80, 64, 200)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    t0 = time.time()
    result = trainer.train_task(task_id=tid, dataloader=dl, epochs=4)
    final = result['final_loss'] if isinstance(result, dict) else result
    drift = result['drift'] if isinstance(result, dict) else 0
    print(f'Task {tid}: loss={final:.4f}, drift={drift:.6f}, time={time.time()-t0:.1f}s')

print()
print('=== 外部记忆验证 ===')
all_loaders = {f'task_{i}': DataLoader(RandDataset(100, 64, 200), batch_size=16, shuffle=False, collate_fn=collate_fn) for i in range(5)}
results = trainer.evaluate_all(dataloaders=all_loaders)
for k, v in results.items():
    print(f'{k} PPL: {v:.2f}')

print()
mem_stats = trainer.get_memory_stats()
print('Memory stats:', mem_stats)
print('OK')
