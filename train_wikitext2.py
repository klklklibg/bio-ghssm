"""WikiText-2 词级 BioGHSSM 训练 - 无外部依赖版"""
import torch, sys, os, time, json
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, '.')

print("=" * 60)
print("WikiText-2 词级 BioGHSSM 训练 (Standalone)")
print("=" * 60)

# ── 用纯 Python 读 PTB char-level 原始文本 ──────────────────────────
import pathlib

# PTB char-level 文本位置
PTB_DIR = pathlib.Path.home() / '.cache' / 'torch' / 'text' / 'wik2'
if not PTB_DIR.exists():
    PTB_DIR = pathlib.Path.home() / '.cache' / 'torch' / 'text'

print(f"\n[1] 扫描 PTB 数据: {PTB_DIR}")
found_files = list(PTB_DIR.rglob('*.txt')) + list(PTB_DIR.rglob('*.pt'))
print(f"  找到 {len(found_files)} 个文件")

# 如果没找到，直接用 char-level PTB 的 train.txt
RAW_PTB = pathlib.Path.home() / '.cache' / 'torch' / 'text' / 'wik2' / 'train.txt'
if not RAW_PTB.exists():
    # 尝试其他位置
    for candidate in [
        pathlib.Path.home() / '.cache' / 'huggingface' / 'datasets' / 'wikitext-2-v1' / 'wikitext-2-v1.train',
        pathlib.Path('D:/data/ptb/train.txt'),
        pathlib.Path('C:/data/ptb/train.txt'),
    ]:
        if candidate.exists():
            RAW_PTB = candidate
            break

print(f"  PTB train: {RAW_PTB} -> {'EXISTS' if RAW_PTB.exists() else 'MISSING'}")

if not RAW_PTB.exists():
    # 用简单的随机 token 序列代替演示
    print("  PTB 文件不存在，用随机数据演示...")
    VOCAB = 1000
    SEQ_LEN = 64
    BATCH = 16

    class RandDS(Dataset):
        def __init__(self, size, seq_len, vocab):
            self.data = [torch.randint(1, vocab, (seq_len+1,)) for _ in range(size)]
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i]

    class RandCollate:
        def __call__(self, batch):
            batch = torch.stack(batch)
            return batch[:, :-1], batch[:, 1:]

    task_loaders = []
    for tid in range(5):
        ds = RandDS(200, SEQ_LEN, VOCAB)
        task_loaders.append(DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=RandCollate()))
        print(f"  Task {tid}: {len(ds)} samples (RANDOM)")

    def make_model():
        from independent.model import BioGHSSMModel
        return BioGHSSMModel(vocab_size=VOCAB, d_model=256, ssm_dim=64, n_layers=2)

    USE_REAL_DATA = False
else:
    print(f"  加载 PTB char-level 数据...")
    VOCAB = 256  # byte-level
    SEQ_LEN = 128
    BATCH = 16

    # 读原始文本
    text = RAW_PTB.read_text(encoding='utf-8')
    chars = list(text)
    # Char to index
    char_to_idx = {c: i+1 for i, c in enumerate(sorted(set(chars)))}
    char_to_idx['<unk>'] = 0
    VOCAB = len(char_to_idx) + 1

    tokens = [char_to_idx.get(c, 0) for c in chars]
    print(f"  {len(chars)} chars -> {len(tokens)} tokens, vocab={VOCAB}")

    class CharDS(Dataset):
        def __init__(self, tokens, seq_len, stride=64):
            self.tokens = tokens
            self.seq_len = seq_len
            self.stride = stride
        def __len__(self):
            return max(0, (len(self.tokens) - self.seq_len - 1) // self.stride)
        def __getitem__(self, idx):
            start = idx * self.stride
            x = torch.tensor(self.tokens[start:start+self.seq_len], dtype=torch.long)
            y = torch.tensor(self.tokens[start+1:start+self.seq_len+1], dtype=torch.long)
            return x, y

    def make_loader(tokens, seq_len, stride, batch_size):
        ds = CharDS(tokens, seq_len, stride)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    # 5 个任务，每个任务 2000 spans，stride=64
    n_task = 5
    total_spans = (len(tokens) - SEQ_LEN - 1) // 64
    spans_per_task = total_spans // n_task
    task_loaders = []
    for tid in range(n_task):
        start = tid * spans_per_task * 64
        end = start + spans_per_task * 64 if tid < n_task - 1 else len(tokens)
        task_tokens = tokens[start:start + spans_per_task * 64 + SEQ_LEN + 1]
        dl = make_loader(task_tokens, SEQ_LEN, 64, BATCH)
        print(f"  Task {tid}: spans={spans_per_task}")
        task_loaders.append(dl)

    USE_REAL_DATA = True

    def make_model():
        from independent.model import BioGHSSMModel
        return BioGHSSMModel(vocab_size=VOCAB, d_model=512, ssm_dim=128, n_layers=8)

# ── 模型 ────────────────────────────────────────────────────────────
print(f"\n[2] 模型")
model = make_model()
print(f"  {model.num_total_params()/1e6:.1f}M 参数")

# ── 训练器 ──────────────────────────────────────────────────────────
print(f"\n[3] 训练器")
from independent.trainer import BioGHSSMTrainer
trainer = BioGHSSMTrainer(model, device='cpu', buffer_size=200,
    wake_lr=5e-4, sleep_lr=5e-5, sleep_steps=40)

# ── 训练循环 ────────────────────────────────────────────────────────
print(f"\n[4] 开始训练 (5 tasks x 3 epochs)")
for tid in range(5):
    dl = task_loaders[tid]
    t0 = time.time()
    result = trainer.train_task(task_id=tid, dataloader=dl, epochs=3)
    elapsed = time.time() - t0
    if isinstance(result, dict):
        final = result.get('final_loss', result)
        drift = result.get('drift', 0)
        wake = result.get('wake_loss', 0)
    else:
        final = result; drift = 0; wake = 0
    print(f"  [Task {tid}] wake={wake:.4f} final={final:.4f} drift={drift:.6f} [{elapsed:.1f}s]")

# ── 最终评估 ────────────────────────────────────────────────────────
print(f"\n[5] 最终评估")
ppls = {}
for tid in range(5):
    ppl = trainer.evaluate_task(tid, task_loaders[tid])
    ppls[f'task_{tid}'] = ppl
avg = sum(ppls.values()) / len(ppls)
ppl_str = ', '.join(f"T{i}={ppls[f'task_{i}']:.2f}" for i in range(5))
print(f"  PPL: {ppl_str}")
print(f"  平均: {avg:.2f}")

os.makedirs('runs', exist_ok=True)
with open('runs/wikitext2_results.json', 'w') as f:
    json.dump({
        'model': f"{model.num_total_params()/1e6:.1f}M",
        'data': 'PTB char-level' if USE_REAL_DATA else 'random',
        'task_ppls': {k: round(v, 2) for k, v in ppls.items()},
        'avg_ppl': round(avg, 2)
    }, f, indent=2)
print("Done!")