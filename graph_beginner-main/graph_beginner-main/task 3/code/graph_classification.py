import argparse
import time
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, ZINC
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from models import GCN, GAT, GraphSAGE, GIN


def get_dataset(name, root=r'D:\py_code\public_datasets\graph_classification\planetoid-master\data'):
    path = os.path.join(root, name)
    if name.startswith('TU'):
        tu_name = name[2:]
        return TUDataset(root=os.path.join(root, 'TUDataset'), name=tu_name, use_node_attr=True)
    elif name == 'ZINC':
        return ZINC(root=os.path.join(root, 'ZINC'), subset=True, split='train')
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_model(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout, pool):
    kwargs = dict(in_channels=in_channels, hidden_channels=hidden_channels,
                  out_channels=out_channels, num_layers=num_layers, dropout=dropout, pool=pool)
    if model_name == 'GCN':
        return GCN(**kwargs)
    elif model_name == 'GAT':
        return GAT(**kwargs)
    elif model_name == 'GraphSAGE':
        return GraphSAGE(**kwargs)
    elif model_name == 'GIN':
        return GIN(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train(model, loader, optimizer, device, task_type):
    model.train()
    total_loss = 0
    count = 0
    for data in loader:
        data = data.to(device)
        x = data.x.float() if data.x.dtype != torch.float else data.x
        optimizer.zero_grad()
        out = model(x, data.edge_index, data.batch)
        if task_type == 'regression':
            loss = F.mse_loss(out.squeeze(), data.y.float())
        else:
            loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.y.size(0)
        count += data.y.size(0)
    return total_loss / count


@torch.no_grad()
def eval_model(model, loader, device, task_type):
    model.eval()
    preds, labels = [], []
    for data in loader:
        data = data.to(device)
        x = data.x.float() if data.x.dtype != torch.float else data.x
        out = model(x, data.edge_index, data.batch)
        if task_type == 'regression':
            preds.append(out.squeeze().cpu())
        else:
            preds.append(out.argmax(dim=1).cpu())
        labels.append(data.y.cpu())
    pred = torch.cat(preds, dim=0)
    label = torch.cat(labels, dim=0)
    if task_type == 'regression':
        mae = (pred - label.float()).abs().mean().item()
        return mae
    else:
        return accuracy_score(label, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TUMUTAG', help='TU{Name} (e.g. TUMUTAG, TUPROTEINS) or ZINC')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT', 'GraphSAGE', 'GIN'])
    parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max', 'min', 'add'])
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    task_type = 'regression' if args.dataset == 'ZINC' else 'classification'
    
    if args.dataset == 'ZINC':
        # ZINC is a PyG built-in dataset and requires an internet connection for
        # the first download. If it is not available locally, switch to a TU
        # dataset (e.g. TUMUTAG) which is already cached.
        zinc_root = os.path.join(r'D:\py_code\public_datasets\graph_classification\planetoid-master\data', 'ZINC')
        train_dataset = ZINC(root=zinc_root, subset=True, split='train')
        val_dataset = ZINC(root=zinc_root, subset=True, split='val')
        test_dataset = ZINC(root=zinc_root, subset=True, split='test')
        out_channels = 1
    else:
        dataset = get_dataset(args.dataset)
        dataset = dataset.shuffle()
        n = len(dataset)
        train_dataset = dataset[:int(0.8 * n)]
        val_dataset = dataset[int(0.8 * n):int(0.9 * n)]
        test_dataset = dataset[int(0.9 * n):]
        out_channels = dataset.num_classes

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    in_channels = train_dataset.num_features
    model = get_model(args.model, in_channels, args.hidden, out_channels, args.num_layers, args.dropout, args.pool)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Dataset: {args.dataset}, Model: {args.model}, Pool: {args.pool}, Layers: {args.num_layers}, LR: {args.lr}")
    best_val = float('inf') if task_type == 'regression' else 0
    best_test = float('inf') if task_type == 'regression' else 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train(model, train_loader, optimizer, device, task_type)
        if epoch % 10 == 0 or epoch == 1:
            val_metric = eval_model(model, val_loader, device, task_type)
            test_metric = eval_model(model, test_loader, device, task_type)
            improved = (val_metric < best_val) if task_type == 'regression' else (val_metric > best_val)
            if improved:
                best_val = val_metric
                best_test = test_metric
            metric_name = 'MAE' if task_type == 'regression' else 'Acc'
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val {metric_name}: {val_metric:.4f}, Test {metric_name}: {test_metric:.4f}")
    elapsed = time.time() - start
    print(f"Best Val: {best_val:.4f}, Best Test: {best_test:.4f}, Time: {elapsed:.2f}s")


if __name__ == '__main__':
    main()
