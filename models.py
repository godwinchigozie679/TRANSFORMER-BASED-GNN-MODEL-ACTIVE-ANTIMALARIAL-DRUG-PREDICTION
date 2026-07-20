import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, GATConv, BatchNorm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def _reshape_descriptors(mol_descriptors, batch_size):
    # mol_descriptors comes in flattened across the batch, put it back
    # into [batch_size, n_descriptors]
    if mol_descriptors.dim() == 2:
        return mol_descriptors
    n = mol_descriptors.numel() // batch_size
    return mol_descriptors.view(batch_size, n)


class DescriptorEncoder(nn.Module):
    # small MLP for the descriptor vector. built lazily since we don't
    # know the descriptor count until the first batch shows up (it
    # varies depending on how feature selection went for this split)
    def __init__(self, hidden_dim, out_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.net = None

    def forward(self, mol_descriptors, batch_size):
        x = _reshape_descriptors(mol_descriptors, batch_size)
        if self.net is None or self.net[0].in_features != x.shape[1]:
            self.net = nn.Sequential(
                nn.Linear(x.shape[1], self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.out_dim),
            ).to(x.device)
        return self.net(x)


class GCNBaseline(nn.Module):
    """Plain GCN, mean+max pooling, descriptor branch concatenated at the end."""

    def __init__(self, node_feature_dim=27, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList([GCNConv(node_feature_dim, hidden_dim)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.descriptor_encoder = DescriptorEncoder(hidden_dim, hidden_dim // 2, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_repr = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        desc_repr = self.descriptor_encoder(data.mol_descriptors, graph_repr.shape[0])

        return torch.sigmoid(self.classifier(torch.cat([graph_repr, desc_repr], dim=1)))


class GATNet(nn.Module):
    """Plain GAT, mean+max pooling, descriptor branch concatenated at
    the end — deliberately kept as simple as GCNBaseline (no edge
    features, no residual connections, no alternating layer types).
    The only real difference from GCN is that each atom's neighbors get
    attention-weighted rather than fixed-weighted, so the model can
    learn which neighboring atoms matter more instead of averaging them
    all equally. If this doesn't clearly beat GCN, that's a legitimate
    finding — attention isn't automatically better, it's just a
    different inductive bias that sometimes helps and sometimes doesn't."""

    def __init__(self, node_feature_dim=27, hidden_dim=128, num_heads=8, num_layers=3, dropout=0.2):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim needs to divide evenly by num_heads"
        self.dropout = dropout

        self.convs = nn.ModuleList([
            GATConv(node_feature_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.descriptor_encoder = DescriptorEncoder(hidden_dim, hidden_dim // 2, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_repr = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        desc_repr = self.descriptor_encoder(data.mol_descriptors, graph_repr.shape[0])

        return torch.sigmoid(self.classifier(torch.cat([graph_repr, desc_repr], dim=1)))


class GraphTransformerNet(nn.Module):
    """No longer one of the three active architectures (GCN / GAT /
    GAT-Transformer replaced it) — left here in case you want it back
    later, but nothing in models.py, run.py, or the notebook builds one
    of these by default anymore.

    TransformerConv layers, residual connections, mean+max+add pooling.
    This one's usually the strongest of the three in practice, though
    that's dataset-dependent — don't take that as gospel without checking
    on your own data."""

    def __init__(self, node_feature_dim=27, hidden_dim=128, num_heads=8, num_layers=4, dropout=0.2):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim needs to divide evenly by num_heads"
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.convs = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers)])

        self.graph_encoder = nn.Sequential(nn.Linear(hidden_dim * 3, 64), nn.ReLU(), nn.Dropout(dropout))
        self.descriptor_encoder = DescriptorEncoder(64, 32, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(96, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = F.relu(norm(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
        graph_repr = self.graph_encoder(pooled)
        desc_repr = self.descriptor_encoder(data.mol_descriptors, graph_repr.shape[0])

        return torch.sigmoid(self.classifier(torch.cat([graph_repr, desc_repr], dim=1)))


class GCNTransformerNet(nn.Module):
    """Alternates GCN and TransformerConv layers, attention-weighted sum
    pooling instead of mean/max — same idea as GATTransformerNet, with
    the GAT layers swapped for plain GCN ones.

    Only the TransformerConv layers here see bond features (bond type,
    aromaticity, ring membership, conjugation, bond order) via
    edge-aware attention — GCNConv can't take a multi-dimensional edge
    feature (only a scalar weight), so the GCN layers stay edge-blind
    the same way GCNBaseline is. That's a structural limitation of
    GCNConv, not a deliberate choice made specifically for this
    architecture."""

    def __init__(self, node_feature_dim=27, edge_feature_dim=5, hidden_dim=128, num_heads=8, num_layers=4, dropout=0.3):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_layers = num_layers
        self.dropout = dropout

        self.gcn_in = GCNConv(node_feature_dim, hidden_dim)

        self.transformer_layers = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers - 1):
            if i % 2 == 0:
                self.transformer_layers.append(
                    TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                                     dropout=dropout, concat=True, edge_dim=edge_feature_dim)
                )
            else:
                self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.attn_score = nn.Linear(hidden_dim, 1)
        self.graph_encoder = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout))
        self.descriptor_encoder = DescriptorEncoder(64, 32, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(96, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)

        x = F.dropout(F.relu(self.gcn_in(x, edge_index)), p=self.dropout, training=self.training)

        t_idx = c_idx = 0
        for i in range(self.num_layers - 1):
            residual = x
            if i % 2 == 0:
                x = self.transformer_layers[t_idx](x, edge_index, edge_attr=edge_attr)
                t_idx += 1
            else:
                x = self.gcn_layers[c_idx](x, edge_index)
                c_idx += 1
            x = F.dropout(F.relu(self.norms[i](x)), p=self.dropout, training=self.training)
            x = x + residual

        weights = torch.sigmoid(self.attn_score(x))
        pooled = global_add_pool(x * weights, batch)
        graph_repr = self.graph_encoder(pooled)
        desc_repr = self.descriptor_encoder(data.mol_descriptors, graph_repr.shape[0])

        return torch.sigmoid(self.classifier(torch.cat([graph_repr, desc_repr], dim=1)))


class GATTransformerNet(nn.Module):
    """Alternates GAT and TransformerConv layers, attention-weighted sum
    pooling instead of mean/max. More parameters than the other two, so
    it's also the one most likely to overfit on a small dataset — keep
    an eye on the train/val gap.

    Unlike GCNBaseline and GraphTransformerNet, this one actually uses
    the bond features (bond type, aromaticity, ring membership,
    conjugation, bond order) that graphs.py computes for every edge —
    both GATConv and TransformerConv support edge-aware attention via
    edge_dim, so the attention weights here are informed by what kind
    of bond connects two atoms, not just which atoms are connected.
    The other two architectures structurally can't use this (GCNConv
    only takes a scalar edge weight, not a feature vector) or just
    don't in the current setup. Worth disclosing explicitly if you're
    comparing architectures in a paper — this isn't a pure "same input,
    different architecture" comparison anymore, GAT-Transformer sees
    more information than the other two. That's a legitimate modeling
    choice, but it's a different question than "which architecture is
    intrinsically better," and a reviewer will want to know which
    question you're actually answering."""

    def __init__(self, node_feature_dim=27, edge_feature_dim=5, hidden_dim=128, num_heads=8, num_layers=4, dropout=0.3):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_layers = num_layers
        self.dropout = dropout

        self.gat_in = GATConv(node_feature_dim, hidden_dim // num_heads, heads=num_heads,
                               dropout=dropout, concat=True, edge_dim=edge_feature_dim)

        self.transformer_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers - 1):
            if i % 2 == 0:
                self.transformer_layers.append(
                    TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                                     dropout=dropout, concat=True, edge_dim=edge_feature_dim)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                            dropout=dropout, concat=True, edge_dim=edge_feature_dim)
                )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.attn_score = nn.Linear(hidden_dim, 1)
        self.graph_encoder = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout))
        self.descriptor_encoder = DescriptorEncoder(64, 32, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(96, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)

        x = F.dropout(F.relu(self.gat_in(x, edge_index, edge_attr=edge_attr)), p=self.dropout, training=self.training)

        t_idx = g_idx = 0
        for i in range(self.num_layers - 1):
            residual = x
            if i % 2 == 0:
                x = self.transformer_layers[t_idx](x, edge_index, edge_attr=edge_attr)
                t_idx += 1
            else:
                x = self.gat_layers[g_idx](x, edge_index, edge_attr=edge_attr)
                g_idx += 1
            x = F.dropout(F.relu(self.norms[i](x)), p=self.dropout, training=self.training)
            x = x + residual

        weights = torch.sigmoid(self.attn_score(x))
        pooled = global_add_pool(x * weights, batch)
        graph_repr = self.graph_encoder(pooled)
        desc_repr = self.descriptor_encoder(data.mol_descriptors, graph_repr.shape[0])

        return torch.sigmoid(self.classifier(torch.cat([graph_repr, desc_repr], dim=1)))


def build_baseline_model(seed=42):
    # kept as the original name/default for backward compatibility —
    # this is just build_random_forest below
    return build_random_forest(seed)


def build_random_forest(seed=42):
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)


def build_logistic_regression(seed=42):
    return LogisticRegression(max_iter=1000, random_state=seed)


def build_linear_svm(seed=42):
    # a plain SVC with an RBF kernel doesn't scale to tens of thousands of
    # rows (training cost grows roughly with the square of the sample
    # count, sometimes worse) — this uses a linear SVM instead, wrapped in
    # Platt-scaling calibration so it can still produce probabilities for
    # AUC, which LinearSVC doesn't give you natively
    base = LinearSVC(max_iter=5000, random_state=seed, dual='auto')
    return CalibratedClassifierCV(base, cv=3, n_jobs=-1)


def build_gradient_boosting(seed=42):
    # HistGradientBoostingClassifier, not the classic GradientBoostingClassifier —
    # the histogram-based version (same idea as LightGBM) is what makes this
    # practical on a dataset this size; the classic one would be painfully slow
    return HistGradientBoostingClassifier(random_state=seed)


def build_knn(seed=42):
    return KNeighborsClassifier(n_neighbors=15, n_jobs=-1)


def build_mlp(seed=42):
    return MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=seed, early_stopping=True)


# descriptor-only, no graph at all — these are the numbers the GNNs need
# to beat for the added architectural complexity to be worth it. random
# forest is the default/original one; the rest exist so the comparison in
# a paper isn't riding on a single classical baseline's particular biases
BASELINE_REGISTRY = {
    'random_forest': build_random_forest,
    'logistic_regression': build_logistic_regression,
    'linear_svm': build_linear_svm,
    'gradient_boosting': build_gradient_boosting,
    'knn': build_knn,
    # mlp deliberately left out of the default set -- its internal
    # early-stopping validation split needs a couple of samples per
    # class, which a small or unusually undersampled fold can fail to
    # provide. build_mlp() above still works fine if you want to add it
    # back: BASELINE_REGISTRY['mlp'] = build_mlp
}
