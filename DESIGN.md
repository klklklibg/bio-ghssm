# Bio-GHSSM 独立模型 — 架构设计

## 核心理念

**不是**「在已有LLM外面包一层wrapper」，而是：
- 模型本身就是 Bio-GHSSM 机制的具体实现
- SSM 状态空间是模型的**核心计算单元**，不是外部框架
- Wake-Sleep 双向缓冲是最小学习单元，固化进架构

## 架构设计原则

1. **SSM-first**：用状态空间模型（而非attention）作为序列建模核心
2. **双缓冲最小化**：每层 W_slow（锚点）+ W_fast（可训练），直接内嵌进SSM Cell
3. **Wake-Sleep 内生**：训练循环本身包含 sleep consolidation，不依赖外部调度
4. **独立推理**：训练完可导出纯 SSM 模型，无需运行时包装

## 模型架构

```
BioGHSSMModel (独立模型)
├── Embedding (vocab → d_model)
├── SSMBlock × N  (核心计算层，每块包含双缓冲权重)
│   ├── SSMCell (状态空间单元)
│   │   ├── W_slow: 冻结锚点参数
│   │   └── W_fast: 可训练参数 (fast weights)
│   └── LayerNorm
├── SSMBlock × N
└── Head (d_model → vocab)

每层参数总量 = 2 × (d_model × ssm_dim)  (W_slow + W_fast)
```

## SSMCell 详细设计

```
输入: x_t (batch, d_model)
状态: h_t (batch, ssm_dim)
输出: y_t (batch, d_model)

公式:
  i_t = sigmoid(x_t @ W_fast.input_gate  + b)
  f_t = sigmoid(x_t @ W_fast.forget_gate  + b)   ← SSM 遗忘门
  g_t = tanh(x_t @ W_fast.cell_gate       + b)
  o_t = sigmoid(x_t @ W_fast.output_gate  + b)
  
  h_t = f_t ⊙ h_{t-1} + i_t ⊙ g_t              ← 状态更新
  y_t = o_t ⊙ tanh(h_t @ W_fast.proj + c)       ← 输出投射

W_slow 仅作锚点，不参与 forward 计算
```

**关键**：W_slow 和 W_fast 是**同一个shape的独立参数**，不是separate copies。

## Wake-Sleep 训练协议

### Wake Phase（任务学习）
```
for batch in task_data:
    # W_fast 学习当前任务
    loss = model.forward(batch)          # W_slow 冻结，W_fast 可训练
    loss.backward()
    optimizer_w_fast.step()
    
    # 存储样本到 episodic buffer
    buffer.add(batch)
```

### Sleep Phase（巩固）
```
# W_slow ← W_slow - lr * grad(L_replay)
for step in range(sleep_steps):
    replay_batch = buffer.sample()
    loss_replay = model.forward(replay_batch)   # W_slow 可训练，W_fast 冻结
    
    # W_slow 的梯度来自 replay loss
    # 注意：这里 W_slow 接收梯度，W_fast 不接收
    loss_replay.backward()
    optimizer_w_slow.step()
```

**核心机制**：Sleep 期的梯度是「历史任务样本在当前模型参数下的 loss 梯度」，驱动 W_slow 向「对所有历史任务都好的参数区域」移动。

## 双缓冲参数结构

```python
class SSMCell(nn.Module):
    def __init__(self, d_model, ssm_dim):
        super().__init__()
        # Slow weights: 冻结锚点，EMA追踪
        self.W_slow = nn.Parameter(
            torch.randn(d_model, ssm_dim) * 0.01
        )
        self.W_slow.requires_grad = False
        
        # Fast weights: 可训练
        self.W_fast = nn.Parameter(
            torch.randn(d_model, ssm_dim) * 0.01
        )
        
        # Forward时用 W = W_slow + W_fast（加法整合）
        # 或用 gate: y = (1-g)*W_slow + g*W_fast
```

## 模型规模

| 规模 | 层数 | d_model | ssm_dim | 参数量 |
|------|------|---------|---------|--------|
| Nano | 4 | 128 | 32 | ~200K |
| Small | 6 | 256 | 64 | ~1.5M |
| Medium | 8 | 512 | 128 | ~10M |

## 推理导出

训练完成后，执行 merge：
```python
W_merged = W_slow + W_fast  # 加法合并
# 保存 W_merged 作为最终模型参数
# 推理时无需 wake-sleep 机制，直接跑 forward
```

## 验证任务

- Character-level PTB (small vocab, 序列建模)
- Sequential MNIST (像素扫描)
- 最终目标：CIFAR-10 图像建模（pixel-wise sequence）

## 实现计划

1. `ssm_cell.py` — SSMCell + 双缓冲参数
2. `bio_model.py` — 堆叠 SSMBlock + Embedding + Head
3. `bio_trainer.py` — Wake-Sleep 训练循环
4. `eval.py` — 持续学习评估
5. 优先跑 Nano 规模验证概念
