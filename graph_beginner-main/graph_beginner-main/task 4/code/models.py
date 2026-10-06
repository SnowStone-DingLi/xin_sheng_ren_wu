import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, gamma=12.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.entity_embedding.weight, -6 / math.sqrt(self.embedding_dim), 6 / math.sqrt(self.embedding_dim))
        nn.init.uniform_(self.relation_embedding.weight, -6 / math.sqrt(self.embedding_dim), 6 / math.sqrt(self.embedding_dim))

    def forward(self, heads, relations, tails, negative_tails=None):
        h = self.entity_embedding(heads)
        r = self.relation_embedding(relations)
        t = self.entity_embedding(tails)
        score = torch.norm(h + r - t, p=1, dim=-1)
        if negative_tails is not None:
            t_neg = self.entity_embedding(negative_tails)
            score_neg = torch.norm(h + r - t_neg, p=1, dim=-1)
            return score, score_neg
        return score


class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, gamma=12.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim * 2)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.entity_embedding.weight, -6 / math.sqrt(self.embedding_dim * 2), 6 / math.sqrt(self.embedding_dim * 2))
        nn.init.uniform_(self.relation_embedding.weight, -6 / math.sqrt(self.embedding_dim), 6 / math.sqrt(self.embedding_dim))

    def forward(self, heads, relations, tails, negative_tails=None):
        h = self.entity_embedding(heads).view(-1, self.embedding_dim, 2)
        t = self.entity_embedding(tails).view(-1, self.embedding_dim, 2)
        r_phase = self.relation_embedding(relations) / (self.embedding_dim ** 0.25)
        r = torch.stack([torch.cos(r_phase), torch.sin(r_phase)], dim=-1)
        h = self._rotate(h, r)
        score = self._score(h, t)
        if negative_tails is not None:
            t_neg = self.entity_embedding(negative_tails).view(-1, self.embedding_dim, 2)
            score_neg = self._score(h, t_neg)
            return score, score_neg
        return score

    @staticmethod
    def _rotate(a, r):
        a_real, a_imag = a[..., 0], a[..., 1]
        r_real, r_imag = r[..., 0], r[..., 1]
        real = a_real * r_real - a_imag * r_imag
        imag = a_real * r_imag + a_imag * r_real
        return torch.stack([real, imag], dim=-1)

    @staticmethod
    def _score(h, t):
        return torch.norm(h - t, p=2, dim=-1).sum(dim=-1)


class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, input_dropout=0.2, hidden_dropout=0.3, feat_dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.conv1 = nn.Conv2d(1, 32, (3, 3), stride=1, padding=0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.filt_h = 10
        self.filt_w = embedding_dim // 10
        fc_length = (self.filt_h - 3 + 1) * (self.filt_w - 3 + 1) * 32
        self.fc = nn.Linear(fc_length, embedding_dim)
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.feat_dropout = nn.Dropout(feat_dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, heads, relations, tails=None, negative_tails=None):
        h = self.entity_embedding(heads).view(-1, 1, self.filt_h, self.filt_w)
        r = self.relation_embedding(relations).view(-1, 1, self.filt_h, self.filt_w)
        x = torch.cat([h, r], dim=2)
        x = self.bn0(x)
        x = self.input_dropout(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feat_dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        x = F.relu(x)
        if tails is not None and negative_tails is not None:
            targets = torch.cat([tails, negative_tails], dim=0)
            all_scores = torch.mm(x, self.entity_embedding.weight.t())
            labels_pos = torch.zeros_like(tails)
            labels_neg = torch.ones_like(negative_tails)
            labels = torch.cat([labels_pos, labels_neg], dim=0)
            scores = all_scores.gather(1, targets.view(-1, 1)).squeeze(1)
            return scores, labels.float()
        scores = torch.mm(x, self.entity_embedding.weight.t())
        return scores
