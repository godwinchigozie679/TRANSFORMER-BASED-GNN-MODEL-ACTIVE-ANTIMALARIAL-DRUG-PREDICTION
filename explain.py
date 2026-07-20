"""
Graph-native explainability.

Five methods, each covering both branches of the hybrid model (the
atom graph and the molecular descriptors): GNNExplainer, Integrated
Gradients, Feature Ablation, Saliency, and SHAP.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from captum.attr import IntegratedGradients, FeatureAblation, Saliency

from graphs import ATOM_FEATURE_NAMES


class _SingleGraphAdapter(nn.Module):
    """torch_geometric's Explainer calls the model as
    model(x, edge_index, **kwargs), which does not match the Data+batch
    interface the model expects. This adapter reconstructs the Data
    object before forwarding. Without it, GNNExplainer fails with an
    unhelpful "unexpected keyword argument" error."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, mol_descriptors=None, edge_attr=None, **kwargs):
        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        data.mol_descriptors = mol_descriptors
        data.edge_attr = edge_attr
        return self.model(data)


def explain_single_molecule(model, graph, device):
    graph = graph.to(device)
    adapter = _SingleGraphAdapter(model)

    explainer = Explainer(
        model=adapter,
        algorithm=GNNExplainer(epochs=200),
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


def explain_molecule_set(model, graphs, device, max_molecules=20):
    """Runs GNNExplainer over a sample of molecules and aggregates the
    results into per-atom-feature importance. Molecules for which the
    optimization fails to converge are skipped rather than aborting
    the run."""
    total_importance = np.zeros(len(ATOM_FEATURE_NAMES))
    total_atoms = 0
    n_failed = 0

    for graph in graphs[:max_molecules]:
        try:
            result = explain_single_molecule(model, graph, device)
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
    """Captum requires a model that accepts only the tensor being
    attributed. This wrapper holds edges, descriptors, and batch
    indices fixed while allowing node features to vary.

    Integrated Gradients and Feature Ablation expand the input
    internally along an interpolation or perturbation axis, so
    forward() may be called with a multiple of the original node
    count. In that case the batch index, edge_index, edge_attr, and
    descriptors must be tiled to match, with indices offset per copy.
    Omitting this produces a RuntimeError from torch_geometric's
    scatter operations that does not clearly indicate the underlying
    cause."""

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

        # mol_descriptors is flattened 1D across the batch -- repeat(k, 1)
        # on a 1D tensor doesn't tile it end to end, it makes k copies of
        # the whole vector, which quietly breaks the batch size downstream
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
    """Each node feature is zeroed individually and the resulting
    change in prediction is measured."""
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
    """Gradient of the output with respect to the input, computed on
    the real graph. The cheapest and noisiest of the methods here;
    treated as a preliminary check rather than a definitive result."""
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
    """SHAP is applied to the 27 atom feature columns rather than to
    individual atoms."""

    def __init__(self, model, single_graph):
        super().__init__()
        self.model = model
        self.single_graph = single_graph

    def forward(self, mask):
        # one pass per row -- SHAP hands these in as separate
        # coalitions, each running through the same fixed graph
        outputs = []
        for row in mask:
            working = self.single_graph.clone()
            working.x = self.single_graph.x * row.unsqueeze(0)
            outputs.append(self.model(working))
        return torch.cat(outputs, dim=0)


def shap_on_batch(model, batch, nsamples='auto'):
    """SHAP over atom feature columns, computed one molecule at a time.
    KernelExplainer estimates Shapley values for the all-ones mask
    (the real molecule) against an all-zeros background, and results
    are averaged across molecules as elsewhere in this module.

    Substantially slower than the gradient-based methods, since this
    involves genuine perturbation sampling per molecule."""
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


# ---------------------------------------------------------------------
# Descriptor branch: the RDKit values (MolWt, LogP, TPSA, ...) that
# get concatenated in before the classifier, the other half of what a
# hybrid model bases its prediction on.
# ---------------------------------------------------------------------

def _reshape_mol_descriptors_2d(batch):
    """mol_descriptors arrives as a flattened 1D tensor across the
    batch and is reshaped to [n_graphs, n_descriptors] before being
    passed to Captum or SHAP."""
    n_graphs = int(batch.batch.max().item()) + 1 if batch.batch is not None else 1
    md = batch.mol_descriptors
    if md.dim() == 2:
        return md
    n = md.numel() // n_graphs
    return md.view(n_graphs, n)


def _extract_single_graph(batch, i):
    """Extracts graph i's nodes and edges from a batched Data object,
    including its own batch vector, remapped node indices, and its own
    slice of mol_descriptors.
"""
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
    """Mirror of _NodeFeatureWrapper: holds the graph fixed and allows
    the descriptor vector to vary instead. The same tiling requirement
    applies when Integrated Gradients expands its input, applied here
    to the other branch."""

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
    """Same method as integrated_gradients_on_batch, applied to the
    descriptor branch. descriptor_names must match the columns used to
    construct this batch, since feature selection is correlation-based
    and varies across splits."""
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
    """Feature Ablation on the descriptor branch: each RDKit
    descriptor is zeroed individually with the graph held fixed."""
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
    """Saliency on the descriptor branch; same limitations as
    saliency_on_batch above."""
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
    """SHAP on the descriptor branch. More straightforward than the
    atom-level version, since descriptors are already a flat vector
    with no topology to account for.

    Same per-molecule procedure as shap_on_batch, with an all-zeros
    background. Slower than the gradient-based methods for the same
    reason."""
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
    """GNNExplainer-style mask learning applied to the descriptor
    branch. Not the torch_geometric class itself, which requires
    graph-structured input; the same underlying principle is used
    instead -- a learnable per-descriptor mask optimized to preserve
    the original prediction while encouraging sparsity.

    One mask is learned per molecule and averaged across molecules. A
    high final value indicates the descriptor was necessary to
    preserve the prediction."""
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