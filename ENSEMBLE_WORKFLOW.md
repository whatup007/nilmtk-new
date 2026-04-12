# 集成学习自动化工作流程指南

## 📌 核心改进

**一键运行**：直接运行 `train_ensemble_model.py`，自动完成数据融合 + 模型训练

## 🚀 快速开始

### 方法1：完全自动化（推荐）

```bash
# 使用默认配置（config_3.yaml）自动训练集成模型
python train_ensemble_model.py --appliance kettle
```

**自动流程**：
1. ✅ 从 `config_3.yaml` 读取模型选取和融合方法
2. ✅ 检测是否存在缓存的融合数据
3. ✅ 如不存在，自动调用两个模型对数据进行预测和融合
4. ✅ 使用融合数据训练第二层 Transformer 模型
5. ✅ 生成完整的评估可视化

### 方法2：手动指定融合数据

```bash
# 如果已经生成过融合数据，可以直接使用
python train_ensemble_model.py \
    --appliance kettle \
    --npz-path ensemble_data/ensemble_dataset_kettle_average.npz
```

## ⚙️ 配置文件 `config_3.yaml`

### 1. 模型选取配置

```yaml
ensemble:
  # 第一层模型1
  model1:
    name: 'seq2point'
    checkpoint: 'runs/exp30/weights/best_metrics.pth'
    alias: 'Seq2Point'

  # 第一层模型2
  model2:
    name: 'seq2point_lstm'
    checkpoint: 'runs/exp33/weights/best_metrics.pth'
    alias: 'LSTM'
```

**自定义方法**：
- 修改 `checkpoint` 路径指向不同的实验
- 修改 `alias` 自定义模型名称

### 2. 融合方法配置

```yaml
ensemble:
  # 融合方法选择
  fusion_method: 'average'  # 'average', 'weighted_average', 'stack'

  # 加权平均的权重（仅用于 weighted_average）
  weights: [0.5, 0.5]

  # 缓存设置
  cache_dir: 'ensemble_data'
  use_cache: True  # 是否使用缓存数据
```

**融合方法对比**：

| 方法 | 说明 | 输出形状 | 性能潜力 |
|------|------|---------|---------|
| `average` | 简单平均 | (N,) | ⭐⭐⭐ |
| `weighted_average` | 加权平均 | (N,) | ⭐⭐⭐ |
| `stack` | 特征堆叠 | (N, 2) | ⭐⭐⭐⭐⭐ |

### 3. 缓存机制

```yaml
ensemble:
  use_cache: True  # 启用缓存，避免重复生成数据
```

**缓存逻辑**：
- 首次运行：自动生成融合数据并缓存
- 后续运行：直接使用缓存数据，大幅提速
- 强制重新生成：设置 `use_cache: False`

## 📊 使用示例

### 示例1：训练 kettle 集成模型（简单平均）

```bash
python train_ensemble_model.py --appliance kettle
```

**输出**：
- 融合数据：`ensemble_data/ensemble_dataset_kettle_average.npz`
- 模型权重：`runs/expN/weights/` (best.pth, best_metrics.pth, last.pth)
- 评估图表：`runs/expN/00-04_kettle_*.png`

### 示例2：尝试不同融合方法

#### 修改 `config_3.yaml`：
```yaml
ensemble:
  fusion_method: 'stack'  # 改为堆叠方法
```

#### 运行训练：
```bash
python train_ensemble_model.py --appliance kettle
```

**结果**：自动生成新的融合数据 `ensemble_dataset_kettle_stack.npz`

### 示例3：使用不同的第一层模型

#### 修改 `config_3.yaml`：
```yaml
ensemble:
  model1:
    checkpoint: 'runs/exp44/weights/best_metrics.pth'  # 使用不同实验
  model2:
    checkpoint: 'runs/exp33/weights/best_metrics.pth'
```

#### 运行训练：
```bash
python train_ensemble_model.py --appliance fridge
```

### 示例4：自定义超参数

```bash
python train_ensemble_model.py \
    --appliance kettle \
    --batch-size 512 \
    --seed 123 \
    --config configs/config_3.yaml
```

## 🔧 高级配置

### 调整权重（加权平均）

**适用场景**：一个模型明显优于另一个

#### 修改 `config_3.yaml`：
```yaml
ensemble:
  fusion_method: 'weighted_average'
  weights: [0.6, 0.4]  # 给模型1更高权重
```

### 调整第二层模型结构

```yaml
model:
  transformer:
    d_model: 48        # 模型维度
    nhead: 4           # 注意力头数
    num_layers: 2      # 层数
    dim_feedforward: 96
    dropout_rate: 0.15
```

**优化建议**：
- 融合任务通常比第一层简单，可用更小的模型
- 推荐先用默认配置，再根据结果调整

## 📈 与旧流程对比

### 旧流程（两步）
```bash
# 步骤1：生成融合数据
python generate_ensemble_dataset.py --appliance kettle --fusion-method average

# 步骤2：训练模型
python train_ensemble_model.py --npz-path ensemble_data/ensemble_dataset_kettle_average.npz
```

### 新流程（一步）
```bash
# 一步完成所有操作
python train_ensemble_model.py --appliance kettle
```

**优势**：
- ✅ 减少命令行参数配置
- ✅ 自动缓存管理
- ✅ 配置集中化（config_3.yaml）
- ✅ 更少出错，更易维护

## 🎯 完整工作流程

```bash
# 1. 训练第一层模型（如果尚未完成）
python train.py --appliance kettle --model seq2point          # → exp30
python train.py --appliance kettle --model seq2point_lstm     # → exp33

# 2. 自动训练第二层集成模型
python train_ensemble_model.py --appliance kettle             # → expN

# 3. 对比结果
# 查看 runs/exp30, runs/exp33, runs/expN 的评估图表
```

## ❓ 常见问题

**Q1: 如何重新生成融合数据？**

A: 修改 `config_3.yaml`，设置 `ensemble.use_cache: False`

**Q2: 如何切换融合方法？**

A: 修改 `config_3.yaml`，更改 `ensemble.fusion_method`

**Q3: 如何使用不同的第一层模型组合？**

A: 修改 `config_3.yaml`，更新 `ensemble.model1.checkpoint` 和 `ensemble.model2.checkpoint`

**Q4: 融合数据保存在哪里？**

A: 默认保存在 `ensemble_data/` 目录，格式：`ensemble_dataset_{appliance}_{method}.npz`

**Q5: 如何验证缓存是否生效？**

A: 查看日志，如显示 "✓ 发现缓存的融合数据" 即为缓存生效

## 📦 输出文件结构

```
runs/expN/
├── config.yaml                        # 实验配置
├── weights/
│   ├── best.pth                       # 最低验证损失模型
│   ├── best_metrics.pth              # 最佳综合指标模型
│   └── last.pth                       # 最后一轮模型
├── logs/
│   ├── training.log                   # 训练日志
│   └── training_history.json          # 训练历史
└── *.png                              # 评估可视化图表

ensemble_data/
├── ensemble_dataset_kettle_average.npz          # 融合数据（简单平均）
├── ensemble_dataset_kettle_weighted_average.npz # 融合数据（加权平均）
├── ensemble_dataset_kettle_stack.npz            # 融合数据（堆叠）
├── predictions_comparison_train_set.png         # 训练集对比
└── predictions_comparison_validation_set.png    # 验证集对比
```

## 🎓 推荐实践

1. **首次运行**：使用默认配置和 `average` 方法
2. **性能对比**：依次尝试三种融合方法，比较效果
3. **模型调优**：根据结果调整第二层模型超参数
4. **结果分析**：对比第一层和第二层的评估图表

---

**提示**：所有配置参数详见 `configs/config_3.yaml` 的详细注释
