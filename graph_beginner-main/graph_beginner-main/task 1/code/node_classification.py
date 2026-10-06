import argparse
import time
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score
from models import GCN, GAT, GraphSAGE, GIN


def get_dataset(name, root=r'D:\py_code\public_datasets\node_classification\planetoid-master\data'):
    path = os.path.join(root, name)
    if name in ['Cora', 'Citeseer']:
        # Planetoid expects root to be the parent directory of the dataset folder.
        return Planetoid(root=root, name=name)
    elif name == 'Flickr':
        return Flickr(root=path)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_model(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout):
    kwargs = dict(in_channels=in_channels, hidden_channels=hidden_channels,
                  out_channels=out_channels, num_layers=num_layers, dropout=dropout)
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


def has_layer_info(batch):
    return hasattr(batch, 'num_sampled_nodes') and batch.num_sampled_nodes is not None


def make_adjs(batch):
    """Convert NeighborLoader batch into list of (edge_index, e_id, size) per layer."""
    edge_index = batch.edge_index
    num_sampled_nodes = batch.num_sampled_nodes
    if hasattr(batch, 'num_sampled_edges') and batch.num_sampled_edges is not None:
        num_sampled_edges = batch.num_sampled_edges
    else:
        n_layers = len(num_sampled_nodes) - 1
        num_sampled_edges = [edge_index.size(1) // n_layers] * n_layers
    adjs = []
    offset = 0
    for i in range(len(num_sampled_nodes) - 1):
        num_dst = num_sampled_nodes[i]
        num_src = num_sampled_nodes[i + 1]
        num_edges = num_sampled_edges[i]
        layer_edge_index = edge_index[:, offset:offset + num_edges]
        offset += num_edges
        size = (num_src, num_dst)
        adjs.append((layer_edge_index, None, size))
    return adjs


def train_full(model, data, optimizer, device):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_full(model, data, device):
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = accuracy_score(data.y[mask].cpu(), pred[mask].cpu())
        accs.append(acc)
    return accs


def train_sample(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    count = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        if has_layer_info(batch):
            adjs = make_adjs(batch)
            out = model.forward_batch(batch.x, adjs)
        else:
            out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
        count += batch.batch_size
    return total_loss / count


@torch.no_grad()
def eval_sample(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        if has_layer_info(batch):
            adjs = make_adjs(batch)
            out = model.forward_batch(batch.x, adjs)
        else:
            out = model(batch.x, batch.edge_index)
        pred = out[:batch.batch_size].argmax(dim=1)
        preds.append(pred.cpu())
        labels.append(batch.y[:batch.batch_size].cpu())
    pred = torch.cat(preds, dim=0)
    label = torch.cat(labels, dim=0)
    return accuracy_score(label, pred)


def run_full(args, data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = 0
    best_test = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_full(model, data, optimizer, device)
        train_acc, val_acc, test_acc = eval_full(model, data, device)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
    elapsed = time.time() - start
    return best_val, best_test, elapsed


def run_sample(args, data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=[args.num_neighbors] * args.num_layers,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    best_val = 0
    best_test = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_sample(model, train_loader, optimizer, device)
        if epoch % 10 == 0 or epoch == 1:
            # Evaluate on the full graph: neighbor sampling at eval time only adds
            # Python-side sampling overhead and does not speed up inference.
            train_acc, val_acc, test_acc = eval_full(model, data, device)
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
    elapsed = time.time() - start
    return best_val, best_test, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Citeseer', 'Flickr'])
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT', 'GraphSAGE', 'GIN'])
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'sample'])
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_neighbors', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = get_dataset(args.dataset)
    data = dataset[0]
    model = get_model(args.model, dataset.num_features, args.hidden, dataset.num_classes, args.num_layers, args.dropout)

    print(f"Dataset: {args.dataset}, Model: {args.model}, Mode: {args.mode}, Layers: {args.num_layers}, LR: {args.lr}")
    if args.mode == 'full':
        val_acc, test_acc, elapsed = run_full(args, data, model)
    else:
        val_acc, test_acc, elapsed = run_sample(args, data, model)
    print(f"Best Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {elapsed:.2f}s")


if __name__ == '__main__':
    main()
