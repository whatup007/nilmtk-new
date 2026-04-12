# 集成学习融合方法详解

## 🎯 三种融合方法对比

### 1. Average（简单平均）⭐⭐⭐

**原理**：对两个模型的预测进行算术平均
```python
fused = (pred1 + pred2) / 2
```

**数据流程**：
- **训练阶段**：
  - 输入：原始聚合功率序列 `(N, 599)`
  - 目标：两个模型预测的平均值 `(N,)`
  - 模型：Transformer（从原始数据学习预测平均结果）
- **测试阶段**：
  - 输入：原始聚合功率序列 `(N, 599)`
  - 输出：直接预测融合结果
  - 目标：真实电器功率 `(N,)`

**特点**：
- ✅ 简单直观，计算量小
- ✅ 适合两个模型性能相近的情况
- ✅ Transformer 学习从原始数据直接预测融合结果
- ⚠️ 权重固定（50:50），无法自适应

**适用场景**：
- 两个第一层模型表现相似
- 快速验证集成学习效果
- 计算资源有限时的首选

---

### 2. Weighted Average（加权平均）⭐⭐⭐

**原理**：根据模型性能分配不同权重
```python
w1, w2 = 0.6, 0.4  # 可调整
fused = w1 * pred1 + w2 * pred2
```

**数据流程**：
- **训练阶段**：
  - 输入：原始聚合功率序列 `(N, 599)`
  - 目标：两个模型预测的加权平均 `(N,)`
  - 模型：Transformer（从原始数据学习预测加权融合结果）
- **测试阶段**：
  - 输入：原始聚合功率序列 `(N, 599)`
  - 输出：直接预测加权融合结果
  - 目标：真实电器功率 `(N,)`

**特点**：
- ✅ 可根据第一层模型性能调整权重
- ✅ 比简单平均更灵活
- ✅ Transformer 学习从原始数据直接预测加权融合结果
- ⚠️ 需要手动设置权重（在 config_3.yaml 中）

**适用场景**：
- 一个第一层模型明显优于另一个
- 需要手动控制模型贡献度
- 例如：Seq2Point MAE=20, LSTM MAE=25 → 设置权重 [0.6, 0.4]

**配置方法**：
```yaml
ensemble:
  fusion_method: 'weighted_average'
  weights: [0.6, 0.4]  # [model1_weight, model2_weight]
```

---

### 3. Stack（特征堆叠）⭐⭐⭐⭐⭐ **推荐**

**原理**：将两个模型的预测作为特征，训练一个简单网络学习最优组合方式
```python
stacked = [pred1, pred2]  # (N, 2)
final = simple_fc_network(stacked)  # (N,)
```

**数据流程**：
- **训练阶段**：
  - 输入：两个模型的预测堆叠 `(N, 2)`
  - 目标：真实电器功率 `(N,)`
  - 模型：简单 FC 网络（3层全连接：2→64→32→1）
  - **学习目标**：从两个预测中学习最优组合权重
- **测试阶段**：
  1. 先用两个第一层模型对原始数据预测 → `(N, 2)`
  2. 将堆叠预测输入第二层网络 → `(N,)`
  3. 与真实值比较

**特点**：
- ✅ 自动学习最优融合策略（不需要手动设置权重）
- ✅ 性能提升潜力最大
- ✅ 可以学习复杂的非线性组合关系
- ✅ 网络结构简单，训练速度快
- ⚠️ 测试时需要调用两个第一层模型（计算量稍大）

**适用场景**：
- 追求最佳性能
- 不确定如何设置权重时
- 两个模型在不同场景下各有优势
- **首选方法**：让模型自动学习最优组合

**网络结构**：
```
Input (2个预测)
    ↓
Linear(2 → 64) + ReLU + Dropout(0.2)
    ↓
Linear(64 → 32) + ReLU + Dropout(0.2)
    ↓
Linear(32 → 1)
    ↓
Output (融合预测)
```

---

## 📊 三种方法的技术对比

| 特性 | Average | Weighted Average | Stack |
|------|---------|------------------|-------|
| **训练输入** | 原始数据 (N, 599) | 原始数据 (N, 599) | 模型预测 (N, 2) |
| **训练目标** | 平均预测 (N,) | 加权平均 (N,) | 真实值 (N,) |
| **第二层模型** | Transformer | Transformer | 简单 FC 网络 |
| **测试流程** | 单次前向 | 单次前向 | 两次预测 + 融合 |
| **权重学习** | 固定 (0.5, 0.5) | 手动设置 | 自动学习 ⭐ |
| **性能潜力** | 中 | 中 | 高 ⭐⭐⭐ |
| **计算复杂度** | 低 | 低 | 中 |
| **推荐优先级** | ★★★ | ★★★ | ★★★★★ |

---

## 🚀 使用示例

### 方法1：Average（默认）

**config_3.yaml**：
```yaml
ensemble:
  fusion_method: 'average'
```

**运行**：
```bash
python train_ensemble_model.py --appliance kettle
```

**输出**：
- 融合数据：`ensemble_data/ensemble_dataset_kettle_average.npz`
- 模型：Transformer（学习从原始数据预测平均结果）

---

### 方法2：Weighted Average

**config_3.yaml**：
```yaml
ensemble:
  fusion_method: 'weighted_average'
  weights: [0.6, 0.4]  # 给模型1更高权重
```

**运行**：
```bash
python train_ensemble_model.py --appliance kettle
```

**输出**：
- 融合数据：`ensemble_data/ensemble_dataset_kettle_weighted_average.npz`
- 模型：Transformer（学习从原始数据预测加权融合结果）

---

### 方法3：Stack（推荐）

**config_3.yaml**：
```yaml
ensemble:
  fusion_method: 'stack'
  use_cache: False  # 首次使用 stack 需要重新生成数据
```

**运行**：
```bash
python train_ensemble_model.py --appliance kettle
```

**输出**：
- 融合数据：`ensemble_data/ensemble_dataset_kettle_stack.npz`
- 模型：简单 FC 网络（自动学习最优组合）

---

## 🎓 推荐实践流程

### 第一步：快速验证（Average）
```bash
# 修改 config_3.yaml: fusion_method: 'average'
python train_ensemble_model.py --appliance kettle
```
**目的**：快速验证集成学习是否有效

### 第二步：性能对比（所有方法）
```bash
# 1. Average
python train_ensemble_model.py --appliance kettle  # fusion_method: average

# 2. Weighted Average（根据第一层模型性能调整权重）
# 修改 config_3.yaml: fusion_method: 'weighted_average', weights: [0.6, 0.4]
python train_ensemble_model.py --appliance kettle

# 3. Stack（推荐）
# 修改 config_3.yaml: fusion_method: 'stack'
python train_ensemble_model.py --appliance kettle
```

### 第三步：分析结果
对比三个方法在测试集上的表现：
- MAE（平均绝对误差）
- RMSE（均方根误差）
- F1 Score（状态识别）
- R²（相关系数）

**预期结果**：Stack > Weighted Average ≥ Average

---

## 💡 常见问题

**Q1: Stack 方法训练时间更长吗？**

A: 不会。Stack 使用简单的 FC 网络（3层），比 Transformer 更快。但测试时需要先运行两个第一层模型。

**Q2: 如何选择权重（Weighted Average）？**

A: 根据第一层模型的 MAE 反比例设置。例如：
- 模型1 MAE=20, 模型2 MAE=30
- 权重 = [1/20, 1/30] 归一化 = [0.6, 0.4]

**Q3: Stack 方法的缓存数据能复用吗？**

A: Stack 和其他方法的数据格式不同，需要单独生成。切换方法时设置 `use_cache: False`。

**Q4: 为什么 Stack 推荐使用简单网络而不是 Transformer？**

A: 因为输入只有2个特征（两个预测），过于复杂的模型容易过拟合。简单的 FC 网络足以学习最优组合。

**Q5: 可以用超过两个第一层模型吗？**

A: 可以！修改 `generate_ensemble_dataset.py` 和 `config_3.yaml` 支持更多模型。Stack 方法自然支持多模型输入。

---

## 📈 性能提升预期

假设第一层模型性能：
- **exp30 (Seq2Point)**: MAE=15.2, F1=0.85
- **exp33 (LSTM)**: MAE=16.8, F1=0.83

**集成学习预期提升**：
- **Average**: MAE=14.5, F1=0.86 (提升 5%)
- **Weighted Average**: MAE=14.2, F1=0.87 (提升 7%)
- **Stack**: MAE=13.5, F1=0.88 (提升 11%) ⭐

---

**总结**：优先尝试 **Stack** 方法，它能自动学习最优融合策略，通常性能最佳！
