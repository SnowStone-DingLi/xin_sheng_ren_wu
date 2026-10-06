import argparse
import time
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score, average_precision_score
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


def decode(z, edge_label_index):
    return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)


def train_full(model, data, optimizer, device):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    out = decode(z, data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(out, data.edge_label.float())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_full(model, data, device):
    model.eval()
    data = data.to(device)
    z = model(data.x, data.edge_index)
    out = decode(z, data.edge_label_index).cpu()
    pred = torch.sigmoid(out)
    label = data.edge_label.cpu()
    auc = roc_auc_score(label, pred)
    ap = average_precision_score(label, pred)
    return auc, ap


def train_sample(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    count = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        if has_layer_info(batch):
            adjs = make_adjs(batch)
            z = model.forward_batch(batch.x, adjs)
        else:
            z = model(batch.x, batch.edge_index)
        out = decode(z, batch.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, batch.edge_label.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.edge_label.size(0)
        count += batch.edge_label.size(0)
    return total_loss / count


@torch.no_grad()
def eval_sample(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        if has_layer_info(batch):
            adjs = make_adjs(batch)
            z = model.forward_batch(batch.x, adjs)
        else:
            z = model(batch.x, batch.edge_index)
        out = torch.sigmoid(decode(z, batch.edge_label_index)).cpu()
        preds.append(out)
        labels.append(batch.edge_label.cpu())
    pred = torch.cat(preds, dim=0)
    label = torch.cat(labels, dim=0)
    auc = roc_auc_score(label, pred)
    ap = average_precision_score(label, pred)
    return auc, ap


def run_full(args, train_data, val_data, test_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_auc = 0
    best_test_auc = 0
    best_test_ap = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_full(model, train_data, optimizer, device)
        if epoch % 10 == 0 or epoch == 1:
            val_auc, val_ap = eval_full(model, val_data, device)
            test_auc, test_ap = eval_full(model, test_data, device)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = test_auc
                best_test_ap = test_ap
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    elapsed = time.time() - start
    return best_val_auc, best_test_auc, best_test_ap, elapsed


def run_sample(args, train_data, val_data, test_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=[args.num_neighbors] * args.num_layers,
        edge_label_index=train_data.edge_label_index,
        edge_label=train_data.edge_label,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    best_val_auc = 0
    best_test_auc = 0
    best_test_ap = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_sample(model, train_loader, optimizer, device)
        if epoch % 10 == 0 or epoch == 1:
            # Evaluate on the full graph to avoid per-edge neighbor sampling overhead.
            val_auc, val_ap = eval_full(model, val_data, device)
            test_auc, test_ap = eval_full(model, test_data, device)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = test_auc
                best_test_ap = test_ap
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    elapsed = time.time() - start
    return best_val_auc, best_test_auc, best_test_ap, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Flickr', choices=['Cora', 'Citeseer', 'Flickr'])
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT', 'GraphSAGE', 'GIN'])
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'sample'])
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_neighbors', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = get_dataset(args.dataset)
    data = dataset[0]
    transform = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, split_labels=False, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)

    model = get_model(args.model, dataset.num_features, args.hidden, args.hidden, args.num_layers, args.dropout)

    print(f"Dataset: {args.dataset}, Model: {args.model}, Mode: {args.mode}, Layers: {args.num_layers}, LR: {args.lr}")
    if args.mode == 'full':
        val_auc, test_auc, test_ap, elapsed = run_full(args, train_data, val_data, test_data, model)
    else:
        val_auc, test_auc, test_ap, elapsed = run_sample(args, train_data, val_data, test_data, model)
    print(f"Best Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}, Time: {elapsed:.2f}s")


if __name__ == '__main__':
    main()
