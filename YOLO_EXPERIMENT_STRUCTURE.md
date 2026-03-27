# 🚀 YOLO风格的实验管理系统

## 概述

模型训练的文件组织方式现已改为类似 **YOLOv5/v8** 的 `runs/expN` 结构。每次训练都会自动创建一个新的实验文件夹，包含所有训练结果、模型和可视化。

---

## 目录结构

```
runs/
├── exp1/
│   ├── weights/
│   │   ├── best.pth              🏆 最佳模型（最低验证损失）
│   │   ├── best_metrics.pth      ⭐ 最佳综合指标模型（最高F1-Score）
│   │   └── last.pth              ⏱️ 最后一epoch的模型
│   ├── logs/
│   │   ├── training_*.log        📝 训练日志
│   │   ├── *_history.json        📊 训练历史数据
│   │   ├── *_history.png         📈 训练曲线
│   │   └── *_metrics.png         📊 指标变化
│   ├── config.yaml               ⚙️ 训练配置参数
│   ├── 01_*_power_error_metrics.png        🔴 功率误差指标图
│   ├── 02_*_state_recognition_metrics.png  ✓ 状态识别指标图
│   ├── 03_*_energy_metrics.png             ⚡ 能量指标图
│   ├── 04_*_relative_error_metrics.png     📉 相对误差指标图
│   └── 05_*_metrics_summary.png            📋 指标汇总表格
│
├── exp2/
│   └── [同exp1结构]
│
└── exp3/
    └── [同exp1结构]
```

---

## 三个模型版本

### 1. **best.pth** - 最低验证损失模型 🏆

- **选择标准**：验证损失最低
- **用途**：主要评估指标，通常用于最终测试
- **特点**：最稳定，泛化能力强

### 2. **best_metrics.pth** - 最佳综合指标模型 ⭐

- **选择标准**：F1-Score最高
- **用途**：状态识别性能优异，适合关注on/off检测
- **特点**：综合考虑精确率和召回率

### 3. **last.pth** - 最后一epoch模型 ⏱️

- **选择标准**：训练最后一个epoch
- **用途**：快速恢复训练，中间检查点
- **特点**：保存最新的训练状态

---

## 训练参数组织

### config.yaml
- 完整的训练配置
- 包含所有超参数
- 便于重现实验

### 可视化文件

| 文件 | 内容 |
|------|------|
| 01_*_power_error_metrics.png | MAE、RMSE、NMAE对比 |
| 02_*_state_recognition_metrics.png | Accuracy、Precision、Recall、F1 |
| 03_*_energy_metrics.png | 能量对比 + 能量准确率仪表 |
| 04_*_relative_error_metrics.png | RAE、RSE、MAPE |
| 05_*_metrics_summary.png | 所有指标汇总表格 |
| *_history.png | 训练/验证损失曲线 |

---

## 使用示例

### 基础训练
```bash
python train.py --config configs/config.yaml
```

**输出**：
```
✓ 实验目录创建: runs/exp1
✓ 配置已保存: runs/exp1/config.yaml
[训练进行中...]
✓ 最后模型已保存: runs/exp1/weights/last.pth
✓ 最佳模型已保存: runs/exp1/weights/best.pth
✓ 可视化图像已保存: runs/exp1/01_*_power_error_metrics.png
...
============================================================
实验目录结构: runs/exp1
============================================================
📁 exp1/
  📁 weights/
    📄 best.pth
    📄 best_metrics.pth
    📄 last.pth
  📁 logs/
    📄 training_*.log
    📄 *_history.json
    📄 *_history.png
    📄 *_metrics.png
  📄 config.yaml
  📄 01_fridge_power_error_metrics.png
  📄 02_fridge_state_recognition_metrics.png
  📄 03_fridge_energy_metrics.png
  📄 04_fridge_relative_error_metrics.png
  📄 05_fridge_metrics_summary.png
============================================================
```

### 带参数的训练
```bash
python train.py \
  --appliance fridge \
  --epochs 100 \
  --batch-size 256 \
  --learning-rate 0.001
```

**输出**：
```
✓ 实验目录创建: runs/exp2
✓ 配置已保存: runs/exp2/config.yaml
[训练进行中...]
```

### 恢复训练
```bash
python train.py \
  --resume runs/exp1/weights/best.pth \
  --epochs 150
```

**说明**：
- 自动加载 exp1 的配置和模型
- 继续训练到150个epoch
- 结果保存到新的 exp3 中

---

## 关键特性

✅ **自动编号** - 自动生成 exp1, exp2, exp3...
✅ **模型管理** - 三个版本自动区分
✅ **参数保存** - 完整的config.yaml备份
✅ **可视化** - 5个独立的评估指标图表
✅ **日志完整** - 训练日志+历史数据
✅ **易于对比** - 不同exp之间独立隔离

---

## 实验对比

对比不同实验的结果：

```bash
# exp1: CNN模型
python train.py --appliance fridge

# exp2: Transformer模型
python train.py --appliance fridge --mode parallel

# exp3: 不同学习率
python train.py --appliance fridge --learning-rate 0.0001
```

三个实验的结果分别保存在：
- `runs/exp1/weights/best.pth` - CNN 最佳
- `runs/exp2/weights/best.pth` - Transformer 最佳
- `runs/exp3/weights/best.pth` - 低学习率 最佳

直接对比 runs 目录下的各个 exp 文件夹即可。

---

## 模型加载示例

```python
import torch

# 加载最佳模型
checkpoint = torch.load('runs/exp1/weights/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 加载最佳综合指标模型
checkpoint = torch.load('runs/exp1/weights/best_metrics.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 加载最后保存的模型
checkpoint = torch.load('runs/exp1/weights/last.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 优势对比

### 之前（checkpoints/）
```
checkpoints/
├── fridge_redd_best.pth
├── fridge_redd_epoch5.pth
├── fridge_redd_epoch10.pth
└── ...  (文件众多，难以管理)
```

### 现在（runs/expN/）
```
runs/exp1/
├── weights/
│   ├── best.pth
│   ├── best_metrics.pth
│   └── last.pth
├── config.yaml
├── 01_*.png
├── 02_*.png
├── 03_*.png
├── 04_*.png
└── 05_*.png
```

**优点**：
- ✅ 清晰的文件组织
- ✅ 完整的配置备份
- ✅ 自动编号的实验
- ✅ 集中的可视化
- ✅ 易于版本管理

---

## 注意事项

1. **旧的 checkpoints/ 目录** 仍然保留，但新训练建议使用 runs/
2. **恢复训练** 时自动创建新的 exp 目录
3. **模型选择**：
   - 用于测试、发表论文 → `best.pth`
   - 用于状态识别任务 → `best_metrics.pth`
   - 快速恢复中断 → `last.pth`

---

## 常见命令

```bash
# 查看所有实验
ls -la runs/

# 查看exp1的权重
ls -la runs/exp1/weights/

# 查看exp1的配置
cat runs/exp1/config.yaml

# 查看exp1的所有可视化
ls -la runs/exp1/*.png

# 加载特定实验的最佳模型用于推理
python inference.py --checkpoint runs/exp3/weights/best.pth
```

---

**最后更新**：2026-02-14
