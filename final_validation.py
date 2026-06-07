"""Bio-GHSSM 最终验证脚本"""
import torch, sys, os, time, json
sys.path.insert(0, '.')
from independent.model import BioGHSSMModel

os.makedirs('runs', exist_ok=True)

print("=" * 60)
print("Bio-GHSSM 最终验证")
print("=" * 60)

# 1. 小模型 Forward + ONNX
print("\n[1] 小模型 Forward + ONNX")
m1 = BioGHSSMModel(vocab_size=100, d_model=256, ssm_dim=64, n_layers=4)
x = torch.randint(1, 100, (4, 128))
y = m1(x)
print(f"  Forward: {x.shape} -> {y.shape}")

t0 = time.time()
for _ in range(5):
    y = m1(x)
print(f"  速度: {(time.time()-t0)/5*1000:.1f}ms/iter")

torch.onnx.export(m1, (x,), 'runs/bio_ghssm_small.onnx',
    input_names=['input_ids'], output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}, 'logits': {0: 'batch', 1: 'seq'}},
    opset_version=17)
size = os.path.getsize('runs/bio_ghssm_small.onnx') / 1024
print(f"  ONNX 导出: {size:.0f}KB")

import onnxruntime as ort
sess = ort.InferenceSession('runs/bio_ghssm_small.onnx', providers=['CPUExecutionProvider'])
out = sess.run(None, {'input_ids': x.numpy().astype('int64')})[0]
print(f"  ONNX Runtime: {out.shape}")

# 2. 中模型
print("\n[2] 中模型 (1B param)")
m2 = BioGHSSMModel(vocab_size=32000, d_model=1024, ssm_dim=256, n_layers=16)
n = m2.num_total_params()
print(f"  参数: {n/1e6:.0f}M")
x2 = torch.randint(1, 32000, (2, 128))
t0 = time.time()
y2 = m2(x2)
print(f"  Forward: {x2.shape} -> {y2.shape}, {(time.time()-t0)*1000:.0f}ms")

# 3. 训练兼容性
print("\n[3] 训练兼容性")
from torch.utils.data import DataLoader, Dataset
from independent.trainer import BioGHSSMTrainer, collate_fn

class RandDataset(Dataset):
    def __init__(self, size, seq_len, vocab):
        self.data = [torch.randint(1, vocab, (seq_len,)) for _ in range(size)]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

m3 = BioGHSSMModel(vocab_size=200, d_model=128, ssm_dim=32, n_layers=2)
trainer = BioGHSSMTrainer(m3, device='cpu', wake_lr=1e-3, sleep_lr=1e-4, sleep_steps=20)
ds = RandDataset(40, 64, 200)
dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
t0 = time.time()
result = trainer.train_task(task_id=0, dataloader=dl, epochs=3)
print(f"  训练: loss={result['final_loss']:.4f}, time={time.time()-t0:.1f}s")
print(f"  Drift: {result['drift']:.6f}")

# 4. 外部记忆多任务
print("\n[4] 外部记忆多任务")
m4 = BioGHSSMModel(vocab_size=200, d_model=128, ssm_dim=32, n_layers=2)
trainer2 = BioGHSSMTrainer(m4, device='cpu', wake_lr=1e-3, sleep_lr=1e-4, sleep_steps=20,
                             use_external_memory=True, memory_alpha=0.25)
for tid in range(3):
    ds = RandDataset(40, 64, 200)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    result = trainer2.train_task(task_id=tid, dataloader=dl, epochs=3)

all_loaders = {f'task_{i}': DataLoader(RandDataset(80, 64, 200), batch_size=16, shuffle=False, collate_fn=collate_fn) for i in range(3)}
ppls = trainer2.evaluate_all(all_loaders)
print(f"  3任务 PPL: " + ", ".join(f"{k.split('_')[1]}={v:.1f}" for k, v in ppls.items()))
print(f"  Memory snapshots: {trainer2.get_memory_stats()['n_snapshots']}")

print("\n" + "=" * 60)
print("全部验证通过!")
print("=" * 60)