"""Bio-GHSSM 持续学习验证脚本"""
import torch, sys, time, json
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

trainer = BioGHSSMTrainer(model, device='cpu', buffer_size=100,
                           wake_lr=1e-3, sleep_lr=1e-4, sleep_steps=30)

# 保存 Task0 的评估集
ds0_eval = RandDataset(100, 64, 200)
dl0_eval = DataLoader(ds0_eval, batch_size=16, shuffle=False, collate_fn=collate_fn)
ppl0_before = trainer.evaluate_task(task_name='task_0', dataloader=dl0_eval)
print(f'Task0 初始 PPL: {ppl0_before:.2f}')

# 3 个任务连续训练
for tid in range(3):
    ds = RandDataset(50, 64, 200)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    t0 = time.time()
    result = trainer.train_task(task_id=tid, dataloader=dl, epochs=3)
    if isinstance(result, dict):
        final = result['final_loss']
        drift = result['drift']
    else:
        final = result
        drift = 0
    print(f'Task {tid}: final_loss={final:.4f}, drift={drift:.6f}, time={time.time()-t0:.1f}s')

print()
print('=== 持续学习验证 ===')
ppl0_after = trainer.evaluate_task(task_name='task_0', dataloader=dl0_eval)
print(f'Task0 训练后 PPL: {ppl0_after:.2f}')
print(f'遗忘 (PPL上升): {ppl0_after - ppl0_before:.2f}')

# 评估所有任务
all_loaders = {f'task_{i}': DataLoader(RandDataset(100, 64, 200), batch_size=16, shuffle=False, collate_fn=collate_fn) for i in range(3)}
results = trainer.evaluate_all(dataloaders=all_loaders)
for k, v in results.items():
    print(f'{k} PPL: {v:.2f}')

print('OK')
