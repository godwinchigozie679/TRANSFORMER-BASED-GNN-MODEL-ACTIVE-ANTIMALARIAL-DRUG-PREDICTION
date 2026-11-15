"""Molecular graphs: the 27 atom features, 5 bond features, scaffolds, RDKit descriptors, and loading your own graph file."""
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


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


_GROUP_MAP = {1: 1, 3: 1, 11: 1, 19: 1, 37: 1, 55: 1, 87: 1, 4: 2, 12: 2, 20: 2, 38: 2, 56: 2, 88: 2,
              5: 13, 13: 13, 31: 13, 49: 13, 81: 13, 6: 14, 14: 14, 32: 14, 50: 14, 82: 14,
              7: 15, 15: 15, 33: 15, 51: 15, 83: 15, 8: 16, 16: 16, 34: 16, 52: 16, 84: 16,
              9: 17, 17: 17, 35: 17, 53: 17, 85: 17, 2: 18, 10: 18, 18: 18, 36: 18, 54: 18, 86: 18}


_HALOGENS = {9, 17, 35, 53, 85}


_COMMON_NONMETALS = {1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53}


_BOND_TYPE = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3,
              Chem.BondType.AROMATIC: 4}


def _period(z):
    for i, cutoff in enumerate(_PERIOD_CUTOFFS, start=2):
        if z <= cutoff:
            return i - 1
    return len(_PERIOD_CUTOFFS) + 1


def _implicit_valence(atom):
    try:
        return atom.GetValence(Chem.ValenceType.IMPLICIT)
    except AttributeError:
        return atom.GetImplicitValence()


def atom_features(atom, mol):
    """27 atom features."""
    idx, z = atom.GetIdx(), atom.GetAtomicNum()
    n_rings = sum(1 for ring in mol.GetRingInfo().AtomRings() if idx in ring)
    return [
        z, atom.GetDegree(), atom.GetTotalDegree(), _implicit_valence(atom),
        int(atom.GetHybridization()), atom.GetFormalCharge(), int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(), atom.GetNumImplicitHs(), atom.GetNumExplicitHs(),
        int(atom.IsInRing()), n_rings,
        int(atom.IsInRingSize(3)), int(atom.IsInRingSize(4)), int(atom.IsInRingSize(5)),
        int(atom.IsInRingSize(6)), int(atom.IsInRingSize(7)), int(atom.IsInRingSize(8)),
        int(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
        atom.GetNumRadicalElectrons(), len(list(atom.GetNeighbors())),
        sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()),
        _period(z), _GROUP_MAP.get(z, 0), int(z in _HALOGENS),
        int(z not in _COMMON_NONMETALS), int(z not in (1, 6)),
    ]


def bond_features(bond):
    """5 bond features: type, aromaticity, ring membership, conjugation, bond order."""
    return [_BOND_TYPE.get(bond.GetBondType(), 0), int(bond.GetIsAromatic()), int(bond.IsInRing()),
            int(bond.GetIsConjugated()), bond.GetBondTypeAsDouble()]


def featurise(smiles, graph=True, desc=True):
    """Graph arrays, Bemis-Murcko scaffold and RDKit descriptors of one molecule (None if invalid).
    graph / desc can be switched off when the user supplies their own graphs / descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = ei = ea = d = None
    if graph:
        x = np.array([atom_features(a, mol) for a in mol.GetAtoms()], dtype=np.float32)
        ei, ea = [], []
        for b in mol.GetBonds():
            i, j, f = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_features(b)
            ei += [[i, j], [j, i]]
            ea += [f, f]
        ei = np.array(ei, dtype=np.int64).T.reshape(2, -1)
        ea = np.array(ea, dtype=np.float32).reshape(-1, 5)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""
    if desc:
        d = []
        for _, fn in Descriptors.descList:
            try:
                d.append(fn(mol))
            except Exception:
                d.append(np.nan)
        d = np.array(d, dtype=np.float64)
    return x, ei, ea, scaffold, d


def load_graph_file(file):
    """Your own graph file: a dict {SMILES: torch_geometric Data} or a list of Data objects with .smiles.
    Each graph needs x (n_atoms x 27), edge_index and edge_attr (n_edges x 5)."""
    obj = torch.load(file, weights_only=False)
    if isinstance(obj, (list, tuple)):
        obj = {d.smiles: d for d in obj}
    assert isinstance(obj, dict), "graph file must be a dict {smiles: Data} or a list of Data with .smiles"
    d0 = next(iter(obj.values()))
    e_dim = d0.edge_attr.shape[1] if getattr(d0, "edge_attr", None) is not None and d0.edge_attr.dim() == 2 else 5
    assert (d0.x.shape[1], e_dim) == (27, 5), f"expected 27 atom / 5 bond features, got {d0.x.shape[1]} / {e_dim}"
    return obj
