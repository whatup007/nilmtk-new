# 🎯 综合评分机制指南

## 概述

模型的"最佳综合指标模型"（`best_metrics.pth`）现已改为基于**多指标综合评分**，而不是仅考虑 F1-Score。

---

## 综合评分公式

综合评分考虑以下五个关键指标：

$$\text{CompositeScore} = w_1 \cdot n_{MAE} + w_2 \cdot n_{RMSE} + w_3 \cdot n_{F1} + w_4 \cdot n_{R^2} + w_5 \cdot n_{EA}$$

其中：

| 组件 | 含义 | 权重 | 说明 |
|------|------|------|------|
| $n_{MAE}$ | 平均绝对误差归一化 | 0.2 | $\frac{1}{1+\text{MAE}}$（越低越好） |
| $n_{RMSE}$ | 均方根误差归一化 | 0.2 | $\frac{1}{1+\text{RMSE}}$（越低越好） |
| $n_{F1}$ | F1-Score | 0.3 | 直接使用，范围[0,1]（越高越好） |
| $n_{R^2}$ | 决定系数 | 0.15 | 直接使用，范围[0,1]（越高越好） |
| $n_{EA}$ | 能量准确度 | 0.15 | 直接使用，范围[0,1]（越高越好） |

---

## 指标详解

### 1. **MAE（平均绝对误差）**
```
MAE = Mean(|y_true - y_pred|)
```
- 衡量**功率预测的平均偏差**
- 单位：瓦（W）
- 越低越好
- 权重：0.2（20%）

### 2. **RMSE（均方根误差）**
```
RMSE = sqrt(Mean((y_true - y_pred)²))
```
- 衡量**大偏差的敏感性**
- 单位：瓦（W）
- 越低越好
- 权重：0.2（20%）

### 3. **F1-Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- 衡量**开/关状态识别精度**
- 范围：[0, 1]
- 越高越好
- 权重：0.3（30%） - **最重的权重**

### 4. **R²（决定系数）**
```
R² = 1 - (SS_res / SS_tot)
SS_res = Sum((y_true - y_pred)²)
SS_tot = Sum((y_true - mean(y_true))²)
```
- 衡量**模型对方差的解释程度**
- 范围：[-∞, 1]（通常[-1, 1]）
- 越高越好
- 权重：0.15（15%）

### 5. **能量准确度（Energy Accuracy）**
```
EA = 1 - |Total_Energy_True - Total_Energy_Pred| / Total_Energy_True
```
- 衡量**总能量预测的准确性**
- 范围：[0, 1]
- 越高越好
- 权重：0.15（15%）

---

## 综合评分范围

综合评分范围：**[0, 1]**

- **0.9-1.0** 🌟 - 优秀模型（方向全部都很好）
- **0.8-0.9** ⭐ - 很好的模型
- **0.7-0.8** ✓ - 不错的模型
- **0.6-0.7** ◐ - 一般的模型
- **< 0.6** ✗ - 需要改进

---

## 为什么使用综合评分？

### 单一指标的问题

| 指标 | 问题 |
|------|------|
| **仅 F1-Score** | 忽视功率预测精度，可能高估某些不准确的模型 |
| **仅 MAE** | 忽视状态识别（开/关），可能倾向于总是预测中间值 |
| **仅 RMSE** | 对异常值过度敏感，可能不稳定 |
| **仅 R²** | 在功率波动小的数据上表现不好 |

### 综合评分的优势

✅ **平衡多个方面** - 同时考虑功率预测和状态识别
✅ **鲁棒性强** - 多指标加权，任何一个异常都不会主导决策
✅ **实际应用性** - 更接近真实的模型评估需要
✅ **可解释性** - 清晰的权重分配和组件分解

---

## 训练过程中的输出

训练时，当发现更好的综合指标模型时，会输出：

```
最佳综合指标！综合评分: 0.8234 (MAE: 12.3456, RMSE: 23.4567, F1: 0.8901, R²: 0.7234, 能量准确度: 0.9012)
```

含义：
- **综合评分**: 0.8234（很好的模型）
- **MAE**: 12.35 W（预测平均偏差）
- **RMSE**: 23.46 W（包含大偏差）
- **F1**: 0.8901（状态识别准确率89%）
- **R²**: 0.7234（解释72%的方差）
- **能量准确度**: 0.9012（总能量误差<10%）

---

## 权重调整

如果需要调整权重（例如，更强调功率准确度），可以在代码中修改：

```python
# utils/metrics.py - compute_composite_score 函数

composite_score = compute_composite_score(
    metrics,
    mae_weight=0.25,        # 增加 MAE 权重
    rmse_weight=0.25,       # 增加 RMSE 权重
    f1_weight=0.25,         # 减少 F1 权重
    r2_weight=0.15,
    energy_weight=0.10
)
```

---

## 模型选择指南

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| **论文发表/精准度最重** | `best.pth` | 最低验证损失 |
| **生产环境部署** | `best_metrics.pth` | 综合性能最好 |
| **快速迭代/实验** | `last.pth` | 保存最后训练状态 |
| **大负载设备识别** | `best_metrics.pth` | 综合评分最高 |
| **微弱信号场景** | `best.pth` | 损失最低更稳定 |

---

## 代码集成

### 1. 计算综合评分

```python
from utils.metrics import compute_composite_score

# 在验证循环中
composite_score = compute_composite_score(val_metrics)
print(f"综合评分: {composite_score:.6f}")
```

### 2. 获取所有指标

```python
# 包含三个模型版本的最佳评分
best_val_loss: float          # best.pth 的验证损失
best_metrics_score: float     # best_metrics.pth 的综合评分
```

### 3. 加载最佳综合指标模型

```python
import torch

checkpoint = torch.load('runs/exp1/weights/best_metrics.pth')
model.load_state_dict(checkpoint['model_state_dict'])
metrics = checkpoint['val_metrics']
print(f"综合评分: {compute_composite_score(metrics):.4f}")
```

---

## 常见问答

**Q: 为什么 F1-Score 权重最高（30%）？**
A: 对于 NILM 任务，准确识别设备的开/关状态比精确的功率值更重要。F1-Score 同时考虑了精确率和召回率。

**Q: 能否取消某些指标的权重？**
A: 可以将某个指标的权重设为0，但不建议这样做。综合评分的力量来自于多维度的平衡。

**Q: 为什么 best.pth 和 best_metrics.pth 通常不同？**
A: `best.pth` 是最低验证损失，`best_metrics.pth` 是综合性能最好。某个模型可能损失低但F1很差，或相反。

**Q: 如何在评估时也使用综合评分？**
A: 在 `evaluate.py` 中调用 `compute_composite_score(test_metrics)` 即可。

---

**最后更新**: 2026-02-14
**版本**: 1.0
