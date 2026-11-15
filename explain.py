

"""Stage 6 (explain): post-hoc, feature-level explanations of a trained model, same five methods as the original
analysis: GNNExplainer, Integrated Gradients, Feature Ablation, Saliency and SHAP, each on both branches of the
model (atomic graph features and molecular descriptors).

Use it like this (each method is one notebook cell):
    models, ctx, fd = explain.load_trained_models()          # loads the saved models from trained_models/
    setup = explain.make_setup(models["gat_transformer"], ctx, fd)
    explain.run_gnnexplainer(setup)                          # -> test_results/*.csv, figures/*.pdf
    explain.run_integrated_gradients(setup)
    explain.run_feature_ablation(setup)
    explain.run_saliency(setup)
    explain.run_shap(setup)
    explain.run_agreement()                                  # how much the five methods agree (reviewer request)
    explain.run_deletion_check(setup)                        # does removing the top-ranked descriptors matter?

The explanations are post-hoc hypotheses about what the model uses. They are not validated experimentally, and
nothing here identifies substructures. The explained compounds are a random, class-balanced sample of the test set.
"""
import itertools
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import FeatureAblation, IntegratedGradients, Saliency
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader

from config import CFG, DEVICE, KINDS, logged, path, pkl_key
from data_prep import Ctx
from descriptors import FoldData
from graphs import ATOM_FEATURE_NAMES
from models import Net


# ----------------------------------------------------------------------------- loading the saved models
class _Probability(nn.Module):
    """Presents a trained Net the way the explainers expect: model(data) -> probability of shape (n_graphs, 1).
    Descriptors come from data.mol_descriptors (already standardised), atom/bond features are raw."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, data):
        d = data.mol_descriptors
        if d.dim() == 1:
            d = d.view(int(data.batch.max().item()) + 1, -1)
        return torch.sigmoid(self.net(data, desc=d)).unsqueeze(-1)


class _Logit(_Probability):
    """Same, but returns the raw logit (what torch_geometric's GNNExplainer expects for return_type='raw')."""

    def forward(self, data):
        d = data.mol_descriptors
        if d.dim() == 1:
            d = d.view(int(data.batch.max().item()) + 1, -1)
        return self.net(data, desc=d).unsqueeze(-1)


def load_trained_models(kinds=KINDS, seed=None):
    """Loads the saved weights from trained_models/ (no retraining). Returns (models, ctx, fd): the networks in
    eval mode, the cached data, and the descriptor/standardisation bundle the models were trained with."""
    ctx = Ctx()
    fd = FoldData(ctx, ctx.inner_train, ctx.inner_val, CFG["seed"])
    models = {}
    for kind in kinds:
        folder = os.path.dirname(path("trained_models", "x"))
        seeds = sorted(int(f.split("_seed")[1].split(".")[0]) for f in os.listdir(folder) if f.startswith(kind + "_seed"))
        assert seeds, f"no trained {kind} model in {folder}; run the final stage first"
        s = seed if seed is not None else seeds[0]
        ck = torch.load(path("trained_models", f"{kind}_seed{s}.pt"), map_location=DEVICE, weights_only=False)
        assert ck["desc_names"] == list(fd.pipe.names), "descriptor set differs from the one used in training"
        net = Net(kind, fd, heads=CFG["heads"], **ck["params"]).to(DEVICE)
        net.load_state_dict(ck["state"], strict=False)
        models[kind] = net.eval()
        print(f"loaded {pkl_key(kind)} (seed {s})")
    return models, ctx, fd


def explain_subset(ctx, n):
    """Random class-balanced sample of test compounds, interleaved active/inactive (no case selection)."""
    rng = np.random.default_rng(CFG["seed"])
    te = np.asarray(ctx.test_idx)
    pos, neg = te[ctx.y[te] == 1], te[ctx.y[te] == 0]
    h = max(n // 2, 1)
    pos, neg = rng.choice(pos, min(h, len(pos)), replace=False), rng.choice(neg, min(h, len(neg)), replace=False)
    out = [i for pair in zip(pos, neg) for i in pair]
    return np.array(out + list(pos[len(neg):]) + list(neg[len(pos):]))


class ExplainSetup:
    """The model to explain, the explained compounds, and the descriptor names."""

    def __init__(self, net, ctx, fd, n):
        self.net, self.n = net, n
        self.model, self.logit_model = _Probability(net).to(DEVICE).eval(), _Logit(net).to(DEVICE).eval()
        self.subset = explain_subset(ctx, n)
        self.graphs = []
        for i in self.subset:
            g = ctx.graphs[i].clone()
            g.mol_descriptors = torch.from_numpy(fd.table[i]).unsqueeze(0)
            self.graphs.append(g)
        self.labels = ctx.y[self.subset]
        self.batch = next(iter(DataLoader(self.graphs, batch_size=len(self.graphs)))).to(DEVICE)
        self.descriptor_names = list(fd.pipe.names)
        self.device = DEVICE


def make_setup(net, ctx, fd, n=None):
    setup = ExplainSetup(net, ctx, fd, n or CFG["explain_n"])
    print(f"explaining {len(setup.graphs)} random test compounds ({int(setup.labels.sum())} active), "
          f"{len(setup.descriptor_names)} descriptors")
    return setup


# ----------------------------------------------------------------------------- the five methods (as in the original analysis)
class _SingleGraphAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, mol_descriptors=None, edge_attr=None, **kwargs):
        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        data.mol_descriptors = mol_descriptors
        data.edge_attr = edge_attr
        return self.model(data)


def explain_single_molecule(model, graph, device, epochs=200):
    graph = graph.to(device)
    adapter = _SingleGraphAdapter(model)

    explainer = Explainer(
        model=adapter,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(mode='binary_classification', task_level='graph', return_type='raw'),
    )

    explanation = explainer(
        graph.x, graph.edge_index,
        mol_descriptors=graph.mol_descriptors, edge_attr=graph.edge_attr,
    )

    return {
        'node_mask': explanation.node_mask.detach().cpu().numpy(),
        'edge_mask': explanation.edge_mask.detach().cpu().numpy(),
        'smiles': getattr(graph, 'smiles', None),
    }


def explain_molecule_set(model, graphs, device, max_molecules=20, epochs=200):
    total_importance = np.zeros(len(ATOM_FEATURE_NAMES))
    total_atoms = 0
    n_failed = 0

    for graph in graphs[:max_molecules]:
        try:
            result = explain_single_molecule(model, graph, device, epochs)
        except Exception:
            n_failed += 1
            continue
        raw_mask = np.abs(result['node_mask'])
        total_importance += raw_mask.sum(axis=0)
        total_atoms += raw_mask.shape[0]

    if n_failed:
        print(f"GNNExplainer failed on {n_failed}/{min(max_molecules, len(graphs))} molecules, skipped those")

    if total_atoms == 0:
        return None

    total_importance /= total_atoms
    return pd.DataFrame({
        'feature': ATOM_FEATURE_NAMES,
        'importance': total_importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)


class _NodeFeatureWrapper(nn.Module):
    def __init__(self, model, batch):
        super().__init__()
        self.model = model
        self.batch = batch
        self.n_nodes = batch.x.shape[0]
        self.n_edges = batch.edge_index.shape[1]
        self.n_graphs = int(batch.batch.max().item()) + 1 if batch.batch is not None else 1

    def forward(self, x):
        n_total = x.shape[0]

        if n_total == self.n_nodes:
            working = self.batch.clone()
            working.x = x
            return self.model(working)

        if n_total % self.n_nodes != 0:
            raise ValueError(
                f"expanded input has {n_total} rows, not a whole multiple of the "
                f"original {self.n_nodes} -- can't tile the graph structure to match"
            )
        k = n_total // self.n_nodes
        device = x.device

        batch_idx = self.batch.batch.repeat(k)
        graph_offsets = torch.arange(k, device=device).repeat_interleave(self.n_nodes) * self.n_graphs
        batch_idx = batch_idx + graph_offsets

        edge_index = self.batch.edge_index.repeat(1, k)
        edge_offsets = torch.arange(k, device=device).repeat_interleave(self.n_edges) * self.n_nodes
        edge_index = edge_index + edge_offsets

        edge_attr = self.batch.edge_attr.repeat(k, 1) if self.batch.edge_attr is not None else None

        if self.batch.mol_descriptors is not None:
            if self.batch.mol_descriptors.dim() == 1:
                mol_descriptors = self.batch.mol_descriptors.repeat(k)
            else:
                mol_descriptors = self.batch.mol_descriptors.repeat(k, 1)
        else:
            mol_descriptors = None

        working = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        working.batch = batch_idx
        working.mol_descriptors = mol_descriptors
        return self.model(working)


def integrated_gradients_on_batch(model, batch, n_steps=50):
    wrapper = _NodeFeatureWrapper(model, batch)
    ig = IntegratedGradients(wrapper)

    inputs = batch.x.clone().detach().requires_grad_(True)
    baseline = torch.zeros_like(inputs)

    attributions = ig.attribute(inputs, baselines=baseline, n_steps=n_steps, internal_batch_size=32)
    importance = attributions.abs().mean(dim=0).detach().cpu().numpy()

    return pd.DataFrame({
        'feature': ATOM_FEATURE_NAMES,
        'importance': importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)


def feature_ablation_on_batch(model, batch):
    wrapper = _NodeFeatureWrapper(model, batch)
    ablator = FeatureAblation(wrapper)

    inputs = batch.x.clone().detach().requires_grad_(True)
    attributions = ablator.attribute(inputs, target=None, perturbations_per_eval=1)
    importance = attributions.abs().mean(dim=0).detach().cpu().numpy()

    return pd.DataFrame({
        'feature': ATOM_FEATURE_NAMES,
        'importance': importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)


def saliency_on_batch(model, batch):
    wrapper = _NodeFeatureWrapper(model, batch)
    saliency = Saliency(wrapper)

    inputs = batch.x.clone().detach().requires_grad_(True)
    attributions = saliency.attribute(inputs, target=None)
    importance = attributions.abs().mean(dim=0).detach().cpu().numpy()

    return pd.DataFrame({
        'feature': ATOM_FEATURE_NAMES,
        'importance': importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)


class _AtomFeatureColumnWrapper(nn.Module):
    def __init__(self, model, single_graph):
        super().__init__()
        self.model = model
        self.single_graph = single_graph

    def forward(self, mask):
        outputs = []
        for row in mask:
            working = self.single_graph.clone()
            working.x = self.single_graph.x * row.unsqueeze(0)
            outputs.append(self.model(working))
        return torch.cat(outputs, dim=0)


def shap_on_batch(model, batch, nsamples='auto'):
    import shap

    device = next(model.parameters()).device
    n_graphs = int(batch.batch.max().item()) + 1 if batch.batch is not None else 1
    n_features = len(ATOM_FEATURE_NAMES)

    per_molecule_importances = []
    for i in range(n_graphs):
        single_graph = _extract_single_graph(batch, i)
        wrapper = _AtomFeatureColumnWrapper(model, single_graph)

        def predict_fn(mask_array, wrapper=wrapper):
            mask_tensor = torch.tensor(mask_array, dtype=torch.float32, device=device)
            with torch.no_grad():
                out = wrapper(mask_tensor)
            return out.detach().cpu().numpy().flatten()

        instance = np.ones((1, n_features))
        background = np.zeros((1, n_features))

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(instance, nsamples=nsamples, silent=True)
        per_molecule_importances.append(np.abs(np.array(shap_values)).flatten())

    importance = np.mean(per_molecule_importances, axis=0)

    return pd.DataFrame({
        'feature': ATOM_FEATURE_NAMES,
        'importance': importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)


def _reshape_mol_descriptors_2d(batch):
    n_graphs = int(batch.batch.max().item()) + 1 if batch.batch is not None else 1
    md = batch.mol_descriptors
    if md.dim() == 2:
        return md
    n = md.numel() // n_graphs
    return md.view(n_graphs, n)


def _extract_single_graph(batch, i):
    device = batch.x.device
    node_mask = batch.batch == i
    node_indices = torch.where(node_mask)[0]

    remap = torch.full((batch.x.shape[0],), -1, dtype=torch.long, device=device)
    remap[node_indices] = torch.arange(len(node_indices), device=device)

    edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]
    edge_index_sub = remap[batch.edge_index[:, edge_mask]]

    x_sub = batch.x[node_mask]
    edge_attr_sub = batch.edge_attr[edge_mask] if getattr(batch, 'edge_attr', None) is not None else None

    single = Data(x=x_sub, edge_index=edge_index_sub, edge_attr=edge_attr_sub)
    single.batch = torch.zeros(x_sub.shape[0], dtype=torch.long, device=device)

    if getattr(batch, 'mol_descriptors', None) is not None:
        single.mol_descriptors = _reshape_mol_descriptors_2d(batch)[i:i + 1]

    return single


class _DescriptorWrapper(nn.Module):
    def __init__(self, model, batch):
        super().__init__()
        self.model = model
        self.batch = batch
        self.n_nodes = batch.x.shape[0]
        self.n_edges = batch.edge_index.shape[1]
        self.n_graphs = int(batch.batch.max().item()) + 1 if batch.batch is not None else 1

    def forward(self, mol_descriptors):
        n_total = mol_descriptors.shape[0]

        if n_total == self.n_graphs:
            working = self.batch.clone()
            working.mol_descriptors = mol_descriptors
            return self.model(working)

        if n_total % self.n_graphs != 0:
            raise ValueError(
                f"expanded descriptor input has {n_total} rows, not a whole "
                f"multiple of the original {self.n_graphs} graphs"
            )
        k = n_total // self.n_graphs
        device = mol_descriptors.device

        x = self.batch.x.repeat(k, 1)

        batch_idx = self.batch.batch.repeat(k)
        graph_offsets = torch.arange(k, device=device).repeat_interleave(self.n_nodes) * self.n_graphs
        batch_idx = batch_idx + graph_offsets

        edge_index = self.batch.edge_index.repeat(1, k)
        edge_offsets = torch.arange(k, device=device).repeat_interleave(self.n_edges) * self.n_nodes
        edge_index = edge_index + edge_offsets

        edge_attr = self.batch.edge_attr.repeat(k, 1) if self.batch.edge_attr is not None else None

        working = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        working.batch = batch_idx
        working.mol_descriptors = mol_descriptors
        return self.model(working)


def integrated_gradients_descriptors(model, batch, descriptor_names, n_steps=50):
    wrapper = _DescriptorWrapper(model, batch)
    ig = IntegratedGradients(wrapper)

    inputs = _reshape_mol_descriptors_2d(batch).clone().detach().requires_grad_(True)
    baseline = torch.zeros_like(inputs)

    attributions = ig.attribute(inputs, baselines=baseline, n_steps=n_steps, internal_batch_size=32)
    importance = attributions.abs().mean(dim=0).detach().cpu().numpy()

    return pd.DataFrame({
        'descriptor': descriptor_names,
        'importance': importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)


def feature_ablation_descriptors(model, batch, descriptor_names):
    wrapper = _DescriptorWrapper(model, batch)
    ablator = FeatureAblation(wrapper)

    inputs = _reshape_mol_descriptors_2d(batch).clone().detach().requires_grad_(True)
    attributions = ablator.attribute(inputs, target=None, perturbations_per_eval=1)
    importance = attributions.abs().mean(dim=0).detach().cpu().numpy()

    return pd.DataFrame({
        'descriptor': descriptor_names,
        'importance': importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)


def saliency_descriptors(model, batch, descriptor_names):
    wrapper = _DescriptorWrapper(model, batch)
    saliency = Saliency(wrapper)

    inputs = _reshape_mol_descriptors_2d(batch).clone().detach().requires_grad_(True)
    attributions = saliency.attribute(inputs, target=None)
    importance = attributions.abs().mean(dim=0).detach().cpu().numpy()

    return pd.DataFrame({
        'descriptor': descriptor_names,
        'importance': importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)


def shap_descriptors(model, batch, descriptor_names, nsamples='auto'):
    import shap

    device = next(model.parameters()).device
    n_graphs = int(batch.batch.max().item()) + 1 if batch.batch is not None else 1
    all_descriptors = _reshape_mol_descriptors_2d(batch).detach().cpu().numpy()

    per_molecule_importances = []
    for i in range(n_graphs):
        single_graph = _extract_single_graph(batch, i)
        wrapper = _DescriptorWrapper(model, single_graph)

        def predict_fn(x, wrapper=wrapper):
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
            with torch.no_grad():
                out = wrapper(x_tensor)
            return out.detach().cpu().numpy().flatten()

        instance = all_descriptors[i:i + 1]
        background = np.zeros_like(instance)

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(instance, nsamples=nsamples, silent=True)
        per_molecule_importances.append(np.abs(np.array(shap_values)).flatten())

    importance = np.mean(per_molecule_importances, axis=0)

    return pd.DataFrame({
        'descriptor': descriptor_names,
        'importance': importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)


def gnnexplainer_descriptors(model, batch, descriptor_names, epochs=100, lr=0.01,
                              size_coeff=0.01, entropy_coeff=0.1):
    device = next(model.parameters()).device
    n_graphs = int(batch.batch.max().item()) + 1 if batch.batch is not None else 1
    all_descriptors = _reshape_mol_descriptors_2d(batch).detach()
    n_desc = all_descriptors.shape[1]

    per_molecule_masks = []
    for i in range(n_graphs):
        single_graph = _extract_single_graph(batch, i)
        wrapper = _DescriptorWrapper(model, single_graph)

        x = all_descriptors[i:i + 1].to(device)
        with torch.no_grad():
            original_pred = wrapper(x)

        mask_logits = torch.zeros(1, n_desc, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([mask_logits], lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()
            mask = torch.sigmoid(mask_logits)
            masked_pred = wrapper(x * mask)

            fidelity_loss = torch.nn.functional.binary_cross_entropy(masked_pred, original_pred.detach())
            size_loss = mask.sum() * size_coeff
            entropy_loss = -(mask * torch.log(mask + 1e-8)
                              + (1 - mask) * torch.log(1 - mask + 1e-8)).mean() * entropy_coeff

            loss = fidelity_loss + size_loss + entropy_loss
            loss.backward()
            optimizer.step()

        final_mask = torch.sigmoid(mask_logits).detach().cpu().numpy().flatten()
        per_molecule_masks.append(final_mask)

    importance = np.mean(per_molecule_masks, axis=0)

    return pd.DataFrame({
        'descriptor': descriptor_names,
        'importance': importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- one function per method, run each in its own cell
METHODS = [("gnnexplainer", "GNNExplainer"), ("integrated_gradients", "Integrated Gradients"),
           ("feature_ablation", "Feature Ablation"), ("saliency", "Saliency"), ("shap", "SHAP")]


def _save_method(setup, name, atomic, desc):
    """Same files as the original analysis: test_results/<name>_importance.csv (atomic),
    test_results/descriptor_<name>_importance.csv (molecular), figures/<name>_atomic_vs_molecular.pdf."""
    import plots
    if atomic is not None:
        atomic.to_csv(path(f"{name}_importance.csv"), index=False)
    desc.to_csv(path(f"descriptor_{name}_importance.csv"), index=False)
    frames = {"Molecular": desc.rename(columns={"descriptor": "name"})}
    if atomic is not None:
        frames = {"Atomic": atomic.rename(columns={"feature": "name"}), **frames}
    plots.set_publication_style()
    plots.save_feature_importance_pdf(frames, output_path=path("figures", f"{name}_atomic_vs_molecular.pdf"),
                                      feature_col="name")
    return atomic, desc


@logged
def run_gnnexplainer(setup):
    atomic = explain_molecule_set(setup.logit_model, setup.graphs, setup.device, max_molecules=setup.n,
                                  epochs=CFG["gnnexplainer_epochs"])
    desc = gnnexplainer_descriptors(setup.model, setup.batch, setup.descriptor_names,
                                    epochs=CFG["gnnexplainer_desc_epochs"])
    return _save_method(setup, "gnnexplainer", atomic, desc)


@logged
def run_integrated_gradients(setup):
    atomic = integrated_gradients_on_batch(setup.model, setup.batch, n_steps=CFG["ig_steps"])
    desc = integrated_gradients_descriptors(setup.model, setup.batch, setup.descriptor_names, n_steps=CFG["ig_steps"])
    return _save_method(setup, "integrated_gradients", atomic, desc)


@logged
def run_feature_ablation(setup):
    atomic = feature_ablation_on_batch(setup.model, setup.batch)
    desc = feature_ablation_descriptors(setup.model, setup.batch, setup.descriptor_names)
    return _save_method(setup, "feature_ablation", atomic, desc)


@logged
def run_saliency(setup):
    atomic = saliency_on_batch(setup.model, setup.batch)
    desc = saliency_descriptors(setup.model, setup.batch, setup.descriptor_names)
    return _save_method(setup, "saliency", atomic, desc)


@logged
def run_shap(setup):
    """Slowest of the five (real perturbation sampling)."""
    atomic = shap_on_batch(setup.model, setup.batch, nsamples=CFG["shap_nsamples"])
    desc = shap_descriptors(setup.model, setup.batch, setup.descriptor_names, nsamples=CFG["shap_nsamples"])
    return _save_method(setup, "shap", atomic, desc)


# ----------------------------------------------------------------------------- how far the methods agree
def _read_importance(prefix, key, col):
    p = path(f"{prefix}{key}_importance.csv")
    return pd.read_csv(p).set_index(col)["importance"] if os.path.exists(p) else None


@logged
def run_agreement():
    """Rank agreement between the five methods (Spearman correlation and top-k overlap), per branch. Reads the
    CSVs written by the run_* functions, so run those first. Agreement is expected to be partial."""
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plots
    plots.set_publication_style()
    rows, mats = [], {}
    for branch, prefix, col, k in [("Atomic", "", "feature", 5), ("Molecular", "descriptor_", "descriptor", 10)]:
        vecs = {label: _read_importance(prefix, key, col) for key, label in METHODS}
        vecs = {l: v for l, v in vecs.items() if v is not None}
        if len(vecs) < 2:
            print(f"{branch}: fewer than two methods have been run, skipped")
            continue
        labels = list(vecs)
        names = sorted(set.intersection(*[set(v.index) for v in vecs.values()]))
        mat = pd.DataFrame(np.nan, index=labels, columns=labels)
        for a, b in itertools.product(labels, labels):
            va, vb = vecs[a].loc[names].values, vecs[b].loc[names].values
            mat.loc[a, b] = spearmanr(va, vb)[0]
            if a < b:
                top = lambda v: set(np.argsort(-v)[:k])
                rows.append(dict(branch=branch, method_a=a, method_b=b, spearman=mat.loc[a, b], top_k=k,
                                 top_k_overlap=len(top(va) & top(vb)) / k))
        mats[branch] = mat
    out = pd.DataFrame(rows)
    out.to_csv(path("explainability_agreement.csv"), index=False)
    if len(out):
        print(out.groupby("branch")[["spearman", "top_k_overlap"]].agg(["mean", "min", "max"]).round(3).to_string())
    if mats:
        fig, axes = plt.subplots(1, len(mats), figsize=(plots.TARGET_WIDTH_DOUBLE_COL, 3.6), squeeze=False)
        colours = {"Atomic": "#4B83B7", "Molecular": "#C94A45"}
        for i, (ax, (branch, mat)) in enumerate(zip(axes.ravel(), mats.items())):
            sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues", vmin=-0.2, vmax=1, square=True, cbar=False,
                        linewidths=0.6, linecolor="white", annot_kws={"fontsize": 7}, ax=ax)
            ax.set_title(f"{'AB'[i]}. {branch}", fontweight="bold", color=colours.get(branch, "black"))
            ax.tick_params(axis="x", rotation=45, width=1.2)
            ax.tick_params(axis="y", width=1.2)
            for lab in ax.get_xticklabels():
                lab.set_ha("right")
        fig.tight_layout()
        plots._finish(fig, "explainability_agreement")
    return out


@logged
def run_deletion_check(setup, ks=(5, 10, 20), n_random=20):
    """Setting the descriptors a method ranks highest to the training mean should change the predictions more
    than setting the same number of random descriptors. Reads the descriptor_<method>_importance.csv files."""
    names = setup.descriptor_names
    index = {n: i for i, n in enumerate(names)}
    with torch.no_grad():
        p0 = setup.model(setup.batch).squeeze(-1)
    rng = np.random.default_rng(CFG["seed"])

    def change(cols):
        b = setup.batch.clone()
        d = b.mol_descriptors.clone()
        d[:, cols] = 0
        b.mol_descriptors = d
        with torch.no_grad():
            return (setup.model(b).squeeze(-1) - p0).abs().mean().item()

    rows = []
    for key, label in METHODS:
        imp = _read_importance("descriptor_", key, "descriptor")
        if imp is None:
            continue
        order = [index[n] for n in imp.sort_values(ascending=False).index if n in index]
        for k in [k for k in ks if k < len(names)]:
            rnd = [change(rng.choice(len(names), k, replace=False).tolist()) for _ in range(n_random)]
            rows.append(dict(method=label, k=k, mean_abs_change_top_k=change(order[:k]),
                             mean_abs_change_random_k=float(np.mean(rnd)), random_sd=float(np.std(rnd))))
    out = pd.DataFrame(rows)
    out.to_csv(path("explainability_deletion_check.csv"), index=False)
    print(out.round(4).to_string(index=False))
    return out


@logged
def run_all(kind="gat_transformer"):
    """Command-line version: load the saved models, then every method, then the agreement and deletion checks."""
    models, ctx, fd = load_trained_models()
    setup = make_setup(models[kind], ctx, fd)
    for run in (run_gnnexplainer, run_integrated_gradients, run_feature_ablation, run_saliency, run_shap):
        run(setup)
    run_agreement()
    run_deletion_check(setup)
