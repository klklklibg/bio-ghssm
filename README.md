# BioGHSSM: A Standalone State Space Model for Continual Learning

**Bio-inspired Gated Hierarchical State Space Model** — 向量化 SSM + 外部记忆 + Wake-Sleep 课程学习

## 核心创新

- **向量化 SSM**：7ms/iter（比 for 循环快 70 倍）
- **外部记忆**：T=5 任务无灾难性遗忘，PPL 稳定在 215-218
- **Wake-Sleep 课程**：drift 稳定在 0.0001~0.0005 量级
- **Char-level PTB**：完整记忆恢复（负遗忘 -116.85 PPL）

## 模型规格

- **0.5M 参数**：d_model=256, ssm_dim=64, n_layers=4
- **ONNX 可部署**：3.1MB，无需 PyTorch 运行时

## 文件结构

```
docs/           论文、独立实验结果、设计文档
model/          细胞定义 + 模型 + 训练器 + ONNX
scripts/        训练脚本 + 测试脚本
requirements.txt
README.md
```

## 快速使用

```bash
pip install -r requirements.txt
python scripts/train_wikitext2.py      # WikiText-2 训练
python scripts/train_ptb.py             # PTB char-level 训练
python scripts/test_continual.py        # 持续学习测试
```

## 实验结果

详见 `docs/RESULTS.md`

| 实验 | 结果 |
|------|------|
| WikiText-2 (5 tasks×3 epochs) | PPL=1059 |
| 外部记忆 (5 tasks) | PPL 稳定 215~218 |
| Char-level PTB | 负遗忘 -116.85 PPL |
| ONNX 导出 | 3.1MB，推理正确 |

## 许可证

MIT
