import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d


def pool_graph(x, batch, pool_type):
    if pool_type == 'avg':
        return global_mean_pool(x, batch)
    elif pool_type == 'max':
        return global_max_pool(x, batch)
    elif pool_type == 'min':
        return -global_max_pool(-x, batch)
    elif pool_type == 'add':
        return global_add_pool(x, batch)
    else:
        raise ValueError(f"Unknown pool type: {pool_type}")


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, pool='avg'):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.pool = pool
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = pool_graph(x, batch, self.pool)
        x = self.lin(x)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, pool='avg', heads=8):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout))
        self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout
        self.pool = pool
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = pool_graph(x, batch, self.pool)
        x = self.lin(x)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, pool='avg'):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.pool = pool
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = pool_graph(x, batch, self.pool)
        x = self.lin(x)
        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, pool='avg'):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        nn1 = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.convs.append(GINConv(nn1))
        for _ in range(num_layers - 2):
            nn_layer = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
            self.convs.append(GINConv(nn_layer))
        nn_out = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.convs.append(GINConv(nn_out))
        self.dropout = dropout
        self.pool = pool
        self.bn = torch.nn.ModuleList([BatchNorm1d(hidden_channels) for _ in range(max(0, num_layers - 1))])
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bn[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = pool_graph(x, batch, self.pool)
        x = self.lin(x)
        return x
