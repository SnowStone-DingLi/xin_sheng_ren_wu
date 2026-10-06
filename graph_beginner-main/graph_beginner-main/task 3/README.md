# 任务三：图分类

本任务基于 PyTorch Geometric 实现 GCN、GAT、GraphSAGE、GIN 四个主流 GNN 模型在 TUDataset 和 ZINC 数据集上的图分类任务，并对比 AvgPooling、MaxPooling、MinPooling 三种池化方法对性能的影响。

## 文件说明

- `code/models.py`：GCN、GAT、GraphSAGE、GIN 模型定义，支持 avg/max/min/add 池化。
- `code/graph_classification.py`：图分类训练与测试脚本。

## 环境依赖

```bash
pip install -r ../requirements.txt
```

## 运行脚本

### 单条运行

```bash
# TUDataset（以 MUTAG 为例）
python code/graph_classification.py --dataset TUMUTAG --model GCN --pool avg --epochs 200 --lr 0.001 --num_layers 3
python code/graph_classification.py --dataset TUMUTAG --model GIN --pool max --epochs 200 --lr 0.001 --num_layers 3

# ZINC（回归任务）
python code/graph_classification.py --dataset ZINC --model GIN --pool avg --epochs 200 --lr 0.001 --num_layers 4
```

### 批量对比池化方法

```bash
python code/run_experiments.py
```

该脚本会遍历所有模型和池化方法，在指定数据集上运行并输出结果到 `results/` 目录。

## 参数说明

- `--dataset`：数据集名称，`TU{Name}` 表示 TUDataset 中的数据集（如 TUMUTAG、TUPROTEINS），或 `ZINC`。
- `--model`：模型名称（GCN/GAT/GraphSAGE/GIN）。
- `--pool`：池化方法（avg/max/min/add）。
- `--num_layers`：GNN 层数。
- `--lr`：学习率。
- `--epochs`：训练轮数。
- `--batch_size`：批次大小。
