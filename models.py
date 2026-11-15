"""The four graph networks (GCN, GAT, GCN-Transformer, GAT-Transformer) with a descriptor branch and one shared classifier head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG
from torch_geometric.nn import GATConv, GCNConv, TransformerConv, global_max_pool, global_mean_pool


# (column of the raw atom features, number of values, embedding size): atomic number, hybridisation, period, group
EMBED = ((0, 120, 16), (4, 10, 4), (22, 10, 4), (23, 20, 6))


class Net(nn.Module):
    """GCN, GAT, GCN-Transformer or GAT-Transformer with a descriptor branch and a shared classifier."""

    def __init__(self, kind, fd, hidden, layers, dropout, pool, heads=8, **_):
        super().__init__()
        self.kind, self.pool, self.drop = kind, pool, dropout
        in_dim, e_dim, d_dim = 27, 5, fd.table.shape[1]
        self.use_emb = bool(CFG["embed_categorical"])
        if self.use_emb:                                      # learned embeddings of the categorical atom features
            self.emb = nn.ModuleList([nn.Embedding(n, d) for _, n, d in EMBED])
            in_dim += sum(d for _, _, d in EMBED)
        self.register_buffer("x_mu", fd.x_stats[0].clone())
        self.register_buffer("x_sd", fd.x_stats[1].clone())
        self.register_buffer("e_mu", fd.e_stats[0].clone())
        self.register_buffer("e_sd", fd.e_stats[1].clone())
        self.register_buffer("desc", fd.table_t, persistent=False)
        self.hybrid = kind.endswith("transformer")
        self.convs, self.norms, self.edge_aware = nn.ModuleList(), nn.ModuleList(), []
        if self.hybrid:
            self.inp = nn.Linear(in_dim, hidden)
        for l in range(layers):
            d_in = hidden if (self.hybrid or l > 0) else in_dim
            if self.hybrid and l % 2 == 1:
                conv, edge = TransformerConv(d_in, hidden // heads, heads=heads, edge_dim=e_dim), True
            elif kind.startswith("gcn"):
                conv, edge = GCNConv(d_in, hidden), False
            else:
                edge = self.hybrid
                conv = GATConv(d_in, hidden // heads, heads=heads, edge_dim=e_dim if edge else None)
            self.convs.append(conv)
            self.edge_aware.append(edge)
            self.norms.append(nn.LayerNorm(hidden) if self.hybrid else nn.BatchNorm1d(hidden))
        g_dim = hidden * (2 if pool == "mean_max" else 1)
        self.desc_enc = nn.Sequential(nn.Linear(d_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, hidden // 2))
        self.cls = nn.Sequential(nn.Linear(g_dim + hidden // 2, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden // 2, 1))

    def graph_repr(self, b, x=None, ea=None):
        """Pooled graph representation; x / ea (raw features) can be supplied for attribution."""
        x = (b.x if x is None else x)
        ea = (b.edge_attr if ea is None else ea)
        raw = x
        x = (x - self.x_mu) / self.x_sd
        ea = (ea - self.e_mu) / self.e_sd
        if self.use_emb:
            x = torch.cat([x] + [e(raw[:, c].round().long().clamp(0, n - 1))
                                 for e, (c, n, _) in zip(self.emb, EMBED)], dim=1)
        if self.hybrid:
            x = self.inp(x)
        for conv, norm, edge in zip(self.convs, self.norms, self.edge_aware):
            h = conv(x, b.edge_index, ea) if edge else conv(x, b.edge_index)
            h = F.dropout(F.relu(norm(h)), self.drop, self.training)
            x = x + h if self.hybrid else h
        g = global_mean_pool(x, b.batch)
        if self.pool == "mean_max":
            g = torch.cat([g, global_max_pool(x, b.batch)], dim=1)
        return g

    def head(self, g, d):
        """Classifier on the graph representation and the (standardised) descriptor vector."""
        return self.cls(torch.cat([g, self.desc_enc(d)], dim=1)).squeeze(-1)

    def forward(self, b, desc=None, x=None, ea=None):
        g = self.graph_repr(b, x, ea)
        return self.head(g, self.desc[b.idx] if desc is None else desc)
