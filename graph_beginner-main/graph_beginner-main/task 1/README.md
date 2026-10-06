# 任务一：节点分类

本任务基于 PyTorch Geometric 实现 GCN、GAT、GraphSAGE、GIN 四个主流 GNN 模型在 Cora、Citeseer、Flickr 数据集上的节点分类，并对比全图训练与 Neighbor Sampling 子图训练的性能和运行时间。

## 文件说明

- `code/models.py`：GCN、GAT、GraphSAGE、GIN 模型定义。
- `code/node_classification.py`：节点分类训练与测试脚本。

## 环境依赖

```bash
pip install -r ../requirements.txt
```

## 运行脚本

### 全图训练

```bash
python code/node_classification.py --dataset Cora --model GCN --mode full --epochs 200 --lr 0.01 --num_layers 2
python code/node_classification.py --dataset Citeseer --model GAT --mode full --epochs 200 --lr 0.005 --num_layers 2
python code/node_classification.py --dataset Flickr --model GraphSAGE --mode full --epochs 100 --lr 0.01 --num_layers 2
```

### 子图采样训练

```bash
python code/node_classification.py --dataset Cora --model GCN --mode sample --epochs 200 --lr 0.01 --num_layers 2 --batch_size 512 --num_neighbors 25
python code/node_classification.py --dataset Citeseer --model GAT --mode sample --epochs 200 --lr 0.005 --num_layers 2 --batch_size 512 --num_neighbors 25
python code/node_classification.py --dataset Flickr --model GraphSAGE --mode sample --epochs 100 --lr 0.01 --num_layers 2 --batch_size 512 --num_neighbors 25
```

### 全图 vs 子图采样对比实验

```bash
python code/run_experiments.py
```

该脚本会对每一组 `(dataset, model, layers, lr)` 分别运行 `full` 和 `sample` 两种模式，自动对比 Test Acc 和运行时间，并生成 `results/task1_comparison_log.txt`。日志中每行标明设置，且不包含 epoch 训练过程。

对比日志示例：

```text
Task 1 Node Classification: Full vs Neighbor Sampling Comparison - 2026-09-06 12:00:00
====================================================================================================
Dataset    Model        Layers  LR      Mode    Test Acc   Time(s)
----------------------------------------------------------------------------------------------------
Cora       GCN          2       0.01    full    0.8234     5.23
Cora       GCN          2       0.01    sample  0.8156     8.45
  -> Sample vs Full: Test Acc delta=-0.0078, Time delta=+3.22s
...
```

## 参数说明

- `--dataset`：数据集名称（Cora/Citeseer/Flickr）。
- `--model`：模型名称（GCN/GAT/GraphSAGE/GIN）。
- `--mode`：训练模式（full/sample）。
- `--num_layers`：GNN 层数。
- `--lr`：学习率。
- `--epochs`：训练轮数。
- `--batch_size`：采样批次大小（仅 sample 模式）。
- `--num_neighbors`：每跳采样邻居数（仅 sample 模式）。
