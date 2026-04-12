# NILM-SeqPoint: 非侵入式负荷监测（NILM）深度学习项目

本项目是一个基于深度学习的非侵入式负荷监测（NILM）框架，主要实现了基于 **Sequence-to-Point (Seq2Point)** 架构的各种模型，并集成了 **Transformer**、**LSTM** 以及 **集成学习（Ensemble Learning）** 等先进技术。本项目源自“大创”研究项目，旨在提供一个高效、灵活且易于扩展的 NILM 实验平台。

## 🌟 核心特性

- **多模型支持**：实现了经典的 Seq2Point (CNN)、Seq2Point-LSTM 以及基于 Transformer 的 Seq2Point 模型。
- **集成学习架构**：支持多种集成策略（Stacking, Average, Weighted Average, Max, Min），提升预测稳定性。
- **灵活配置**：通过 YAML 配置文件管理所有超参数，支持命令行覆盖。
- **完整流水线**：涵盖从数据预处理、模型训练、性能评估到推理预测的全流程。
- **实验管理**：自动记录训练日志、保存模型权重并生成可视化图表。
- **多数据集支持**：适配 REDD、UK-DALE 等主流 NILM 数据集。

## 📂 项目结构

```text
nilm-seqpoint/
├── configs/                # 配置文件目录
│   ├── config.yaml         # 主配置文件（默认）
│   ├── config_lstm.yaml    # LSTM 模型专用配置
│   └── ...                 # 其他模型配置
├── models/                 # 模型定义
│   ├── seq2point.py        # 经典 CNN Seq2Point 模型
│   ├── seq2point_lstm.py   # LSTM 增强型模型
│   └── seq2point_transformer.py # 基于 Transformer 的模型
├── scripts/                # 工具脚本
│   ├── preprocess_data.py  # 数据预处理（NILMTK 格式转换）
│   ├── check_sampling.py   # 检查数据采样频率
│   └── inspect_data.py     # 数据探索性分析
├── utils/                  # 工具类
│   ├── data_loader.py      # 高效的数据加载与批处理
│   ├── metrics.py          # NILM 专业评估指标（MAE, MSE, SAE, Accuracy 等）
│   └── logger.py           # 训练日志记录
├── train.py                # 主训练入口
├── evaluate.py             # 模型评估入口
├── inference.py            # 推理与可视化脚本
├── generate_ensemble_dataset.py # 生成集成学习数据集
├── train_ensemble_model.py # 训练集成学习元模型
├── requirements.txt        # 依赖清单
└── README.md               # 项目说明文档
```

## 🚀 快速开始

### 1. 环境安装

建议使用 Python 3.9+ 环境。

```bash
pip install -r requirements.txt
```

*注意：如果需要处理原始数据集，请确保已安装 [NILMTK](https://github.com/nilmtk/nilmtk)。*

### 2. 数据准备

将您的数据集（如 REDD）转换为 H5 格式，并放置在项目指定目录下（默认为 `data/`）。

使用预处理脚本：
```bash
python scripts/preprocess_data.py --input_path /path/to/redd.h5 --output_path data/processed_redd.h5
```

### 3. 模型训练

直接运行 `train.py` 开始训练，默认使用 `configs/config.yaml`：

```bash
python train.py --config configs/config.yaml --appliance fridge
```

### 4. 模型评估

评估已训练模型的性能：

```bash
python evaluate.py --model_path results/exp_xxx/weights/best.pth --config configs/config.yaml
```

### 5. 推理与可视化

生成预测结果对比图：

```bash
python inference.py --model_path results/exp_xxx/weights/best.pth --data_path data/test.npz
```

## 🧠 模型架构说明

### Seq2Point (CNN)
本项目实现的核心模型，通过 1D 卷积层提取聚合功率序列的局部特征，最终输出窗口中心点的电器功率。
- **输入**: 长度为 599 的聚合功率窗口。
- **输出**: 单个时间点的电器预测功率。

### Transformer 扩展
利用自注意力机制捕获长距离依赖，适用于具有复杂运行模式的电器。

### 集成学习 (Ensemble)
通过组合多个基模型（如不同架构或不同初始化）的预测结果，采用 Stacking 等方法进一步优化精度。详细说明请参考 [ENSEMBLE_WORKFLOW.md](file:///e:/nilm-seqpoint/ENSEMBLE_WORKFLOW.md)。

## 📊 评估指标

本项目采用 NILM 领域公认的指标进行评估：
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **SAE** (Signal Aggregate Error): 评估一段时间内总能量消耗的预测准确度。
- **Accuracy**: 基于功率阈值的开关状态识别准确率。

## 🛠️ 配置项说明

在 `config.yaml` 中，您可以自定义以下关键参数：
- `data.window_size`: 输入窗口长度（推荐 599）。
- `training.batch_size`: 训练批大小（显存不足时请调小）。
- `training.learning_rate`: 初始学习率。
- `model.name`: 选择使用的模型架构。

## 📝 许可证

本项目遵循 [MIT License](file:///e:/nilm-seqpoint/LICENSE)。

---
*本项目由大创团队维护，欢迎在 Issue 中提出改进建议。*
