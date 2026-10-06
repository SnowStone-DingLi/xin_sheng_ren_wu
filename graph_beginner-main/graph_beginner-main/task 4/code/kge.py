import argparse
import time
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import FB15k_237
from torch.utils.data import DataLoader
from collections import defaultdict
from models import TransE, RotatE, ConvE


def get_dataset(root=r'D:\py_code\public_datasets\knowledge_graph'):
    return FB15k_237(root=os.path.join(root, 'FB15k-237'))


def get_model(model_name, num_entities, num_relations, embedding_dim):
    if model_name == 'TransE':
        return TransE(num_entities, num_relations, embedding_dim)
    elif model_name == 'RotatE':
        return RotatE(num_entities, num_relations, embedding_dim)
    elif model_name == 'ConvE':
        return ConvE(num_entities, num_relations, embedding_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def negative_sampling(heads, relations, tails, num_entities):
    negative_tails = torch.randint(0, num_entities, tails.size(), device=tails.device)
    mask = negative_tails == tails
    while mask.any():
        negative_tails[mask] = torch.randint(0, num_entities, (mask.sum().item(),), device=tails.device)
        mask = negative_tails == tails
    return negative_tails


def train_transe_rotate(model, data, optimizer, device, num_entities):
    model.train()
    heads = data.edge_index[0].to(device)
    relations = data.edge_type.to(device)
    tails = data.edge_index[1].to(device)
    neg_tails = negative_sampling(heads, relations, tails, num_entities)
    pos_score, neg_score = model(heads, relations, tails, neg_tails)
    loss = F.margin_ranking_loss(
        pos_score, neg_score,
        target=torch.ones_like(pos_score),
        margin=6.0
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_conve(model, data, optimizer, device, num_entities):
    model.train()
    heads = data.edge_index[0].to(device)
    relations = data.edge_type.to(device)
    tails = data.edge_index[1].to(device)
    neg_tails = negative_sampling(heads, relations, tails, num_entities)
    scores, labels = model(heads, relations, tails, neg_tails)
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, triples, num_entities, device, max_eval=5000):
    model.eval()
    ranks = []
    triples = triples[:max_eval]
    for h, r, t in triples:
        h_tensor = torch.tensor([h], device=device)
        r_tensor = torch.tensor([r], device=device)
        all_t = torch.arange(num_entities, device=device)
        if isinstance(model, ConvE):
            scores = model(h_tensor, r_tensor).squeeze(0)
        else:
            h_tensor = h_tensor.repeat(num_entities)
            r_tensor = r_tensor.repeat(num_entities)
            scores = model(h_tensor, r_tensor, all_t)
        target_score = scores[t].item()
        rank = (scores > target_score).sum().item() + 1
        ranks.append(rank)
    ranks = torch.tensor(ranks, dtype=torch.float)
    mrr = (1.0 / ranks).mean().item()
    hits1 = (ranks <= 1).float().mean().item()
    hits3 = (ranks <= 3).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()
    return mrr, hits1, hits3, hits10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TransE', choices=['TransE', 'RotatE', 'ConvE'])
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dataset = get_dataset()
    train_data = dataset[0]
    num_entities = train_data.num_nodes
    num_relations = int(train_data.edge_type.max().item()) + 1

    model = get_model(args.model, num_entities, num_relations, args.embedding_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Model: {args.model}, Entities: {num_entities}, Relations: {num_relations}, Dim: {args.embedding_dim}")
    best_mrr = 0
    best_hits1 = best_hits3 = best_hits10 = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        if args.model == 'ConvE':
            loss = train_conve(model, train_data, optimizer, device, num_entities)
        else:
            loss = train_transe_rotate(model, train_data, optimizer, device, num_entities)
        if epoch % 20 == 0 or epoch == 1:
            val_triples = torch.stack([train_data.edge_index[0], train_data.edge_type, train_data.edge_index[1]], dim=1)
            mrr, hits1, hits3, hits10 = evaluate(model, val_triples, num_entities, device, max_eval=2000)
            if mrr > best_mrr:
                best_mrr = mrr
                best_hits1, best_hits3, best_hits10 = hits1, hits3, hits10
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, MRR: {mrr:.4f}, Hits@1: {hits1:.4f}, Hits@3: {hits3:.4f}, Hits@10: {hits10:.4f}")
    elapsed = time.time() - start
    print(f"Best MRR: {best_mrr:.4f}, Hits@1: {best_hits1:.4f}, Hits@3: {best_hits3:.4f}, Hits@10: {best_hits10:.4f}, Time: {elapsed:.2f}s")


if __name__ == '__main__':
    main()
