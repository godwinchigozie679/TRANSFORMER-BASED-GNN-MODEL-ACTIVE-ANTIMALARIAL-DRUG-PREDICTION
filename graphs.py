import os

import torch
import pandas as pd
from torch_geometric.data import Data
from rdkit import Chem

import config as config_module

ATOM_FEATURE_NAMES = [
    'Atomic Number', 'Degree', 'Total Degree', 'Implicit Valence', 'Hybridization',
    'Formal Charge', 'Is Aromatic', 'Total H Count', 'Num Implicit Hs', 'Num Explicit Hs',
    'Is In Ring', 'Num Rings',
    'In 3-member Ring', 'In 4-member Ring', 'In 5-member Ring',
    'In 6-member Ring', 'In 7-member Ring', 'In 8-member Ring',
    'Is Chiral', 'Num Radical Electrons', 'Num Neighbors', 'Sum Bond Orders',
    'Period', 'Group', 'Is Halogen', 'Is Metal', 'Is Heteroatom',
]

BOND_FEATURE_NAMES = ['Bond Type', 'Is Aromatic Bond', 'Bond In Ring', 'Is Conjugated', 'Bond Order']

_PERIOD_CUTOFFS = [2, 10, 18, 36, 54, 86]
_GROUP_MAP = {
    1: 1, 3: 1, 11: 1, 19: 1, 37: 1, 55: 1, 87: 1,
    4: 2, 12: 2, 20: 2, 38: 2, 56: 2, 88: 2,
    5: 13, 13: 13, 31: 13, 49: 13, 81: 13,
    6: 14, 14: 14, 32: 14, 50: 14, 82: 14,
    7: 15, 15: 15, 33: 15, 51: 15, 83: 15,
    8: 16, 16: 16, 34: 16, 52: 16, 84: 16,
    9: 17, 17: 17, 35: 17, 53: 17, 85: 17,
    2: 18, 10: 18, 18: 18, 36: 18, 54: 18, 86: 18,
}
_HALOGENS = {9, 17, 35, 53, 85}
_COMMON_NONMETALS = {1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53}  # H, C, N, O + halogens + P, S, Se


def _period(atomic_num):
    for i, cutoff in enumerate(_PERIOD_CUTOFFS, start=2):
        if atomic_num <= cutoff:
            return i - 1
    return len(_PERIOD_CUTOFFS) + 1


def atom_features(atom, mol):
    ring_info = mol.GetRingInfo()
    idx = atom.GetIdx()
    num_rings = sum(1 for ring in ring_info.AtomRings() if idx in ring)
    z = atom.GetAtomicNum()

    return [
        z,
        atom.GetDegree(),
        atom.GetTotalDegree(),
        atom.GetImplicitValence(),
        int(atom.GetHybridization()),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        atom.GetNumImplicitHs(),
        atom.GetNumExplicitHs(),
        int(atom.IsInRing()),
        num_rings,
        int(atom.IsInRingSize(3)),
        int(atom.IsInRingSize(4)),
        int(atom.IsInRingSize(5)),
        int(atom.IsInRingSize(6)),
        int(atom.IsInRingSize(7)),
        int(atom.IsInRingSize(8)),
        int(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
        atom.GetNumRadicalElectrons(),
        len(list(atom.GetNeighbors())),
        sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds()),
        _period(z),
        _GROUP_MAP.get(z, 0),
        int(z in _HALOGENS),
        int(z not in _COMMON_NONMETALS),  # rough metal flag, fine for organics
        int(z not in (1, 6)),
    ]


def bond_features(bond):
    bond_type_map = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    return [
        bond_type_map.get(bond.GetBondType(), 0),
        int(bond.GetIsAromatic()),
        int(bond.IsInRing()),
        int(bond.GetIsConjugated()),
        bond.GetBondTypeAsDouble(),
    ]


def smiles_to_graph(smiles, descriptor_values=None, graph_idx=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a, mol) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]  # both directions, PyG wants undirected graphs this way
        feat = bond_features(bond)
        edge_attr += [feat, feat]

    if edge_index:
        edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)
    else:
        # single-atom molecules, no bonds at all
        edge_index_t = torch.empty((2, 0), dtype=torch.long)
        edge_attr_t = torch.empty((0, 5))

    data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)
    data.smiles = smiles
    if graph_idx is not None:
        data.graph_idx = graph_idx
    if descriptor_values is not None:
        cleaned = [0.0 if pd.isna(v) else float(v) for v in descriptor_values]
        data.mol_descriptors = torch.tensor(cleaned, dtype=torch.float)

    return data


def build_graph_dataset(df, descriptor_cols, smiles_col='Smiles', label_col='Labels',
                         cache_path=None, use_cache=True, autosave=True):
    """df's descriptor columns should already be scaled -- this just
    packages things into graphs, doesn't fit anything itself.

    Turning SMILES into atoms/bonds/features is the slow part, and it's
    identical across folds and models -- only the descriptors and label
    change per row. Caches that base structure by SMILES to cache_path
    (defaults to config.GRAPH_CACHE_PATH) so a full CV run doesn't
    rebuild the same molecules over and over."""
    cache_path = cache_path or config_module.GRAPH_CACHE_PATH
    graphs = []

    for i, row in df.reset_index(drop=True).iterrows():
        smiles = row[smiles_col]
        base = _get_base_graph(smiles, cache_path) if use_cache else smiles_to_graph(smiles)
        if base is None:
            continue

        graph = Data(
            x=base.x,
            edge_index=base.edge_index,
            edge_attr=base.edge_attr if hasattr(base, 'edge_attr') else None,
        )
        graph.smiles = smiles
        graph.graph_idx = i

        descriptor_values = row[descriptor_cols].tolist()
        cleaned = [0.0 if pd.isna(v) else float(v) for v in descriptor_values]
        graph.mol_descriptors = torch.tensor(cleaned, dtype=torch.float)

        if label_col in row:
            graph.y = torch.tensor([int(row[label_col])], dtype=torch.long)

        graphs.append(graph)

    if use_cache and autosave:
        save_graph_cache(cache_path)

    print(f"built {len(graphs)}/{len(df)} graphs")
    return graphs


# in-memory cache, loaded once and kept warm across calls -- reloading
# from disk every time would defeat the point
_graph_cache = None
_graph_cache_path_loaded = None
_graph_cache_dirty = False


def _load_graph_cache(cache_path):
    global _graph_cache, _graph_cache_path_loaded
    if _graph_cache is not None and _graph_cache_path_loaded == cache_path:
        return

    if os.path.exists(cache_path):
        print(f"loading cached molecular graphs from {cache_path}")
        _graph_cache = torch.load(cache_path, weights_only=False)
        print(f"{len(_graph_cache)} unique structures already cached")
    else:
        _graph_cache = {}
    _graph_cache_path_loaded = cache_path


def _get_base_graph(smiles, cache_path):
    """Cached atom/bond structure for smiles, computed and stored the
    first time we see it. No descriptors or label attached -- those get
    added per row in build_graph_dataset since they change across
    folds even when the molecule doesn't."""
    global _graph_cache_dirty
    _load_graph_cache(cache_path)

    if smiles in _graph_cache:
        return _graph_cache[smiles]

    base = smiles_to_graph(smiles)
    if base is not None:
        _graph_cache[smiles] = base
        _graph_cache_dirty = True
    return base


def save_graph_cache(cache_path=None):
    """Flushes new cache entries to disk. build_graph_dataset already
    calls this automatically, so you don't usually need to."""
    global _graph_cache_dirty
    cache_path = cache_path or config_module.GRAPH_CACHE_PATH
    if _graph_cache is not None and _graph_cache_dirty:
        torch.save(_graph_cache, cache_path)
        print(f"saved {len(_graph_cache)} cached molecular graphs to {cache_path}")
        _graph_cache_dirty = False