# 任务四：知识图谱补全

本任务基于 PyTorch Geometric 实现知识图谱补全模型 TransE、RotatE、ConvE，使用 FB15k-237 数据集进行训练与评估。

## 文件说明

- `code/models.py`：TransE、RotatE、ConvE 模型定义。
- `code/kge.py`：知识图谱补全训练与测试脚本。

## 环境依赖

```bash
pip install -r ../requirements.txt
```

## 运行脚本

### 单模型训练

```bash
python code/kge.py --model TransE --embedding_dim 200 --epochs 200 --lr 0.001
python code/kge.py --model RotatE --embedding_dim 200 --epochs 200 --lr 0.001
python code/kge.py --model ConvE --embedding_dim 200 --epochs 200 --lr 0.001
```

### 批量对比实验

```bash
python code/run_experiments.py
```

该脚本会遍历 TransE、RotatE、ConvE 三个模型，输出结果到 `results/` 目录。

## 参数说明

- `--model`：模型名称（TransE/RotatE/ConvE）。
- `--embedding_dim`：实体/关系嵌入维度。
- `--lr`：学习率。
- `--epochs`：训练轮数。

## 评估指标

- MRR（Mean Reciprocal Rank）
- Hits@1 / Hits@3 / Hits@10
