"""
Draws molecules with RDKit and prints their descriptor values next to
them.

Note: `structural_label` gives a rough heuristic tag (ring/atom
counts), not a real chemical name — don't call it that in a caption.
"""

import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import display


def structural_label(smiles):
    """Quick heuristic label built from ring/heteroatom/halogen counts
    — fine for telling molecules apart at a glance in a grid, not a
    stand-in for an actual name or a real substructure match."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid_structure"

    num_carbons = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C')
    num_nitrogens = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'N')
    num_oxygens = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'O')
    num_rings = mol.GetRingInfo().NumRings()
    num_aromatic_rings = sum(
        1 for ring in mol.GetRingInfo().AtomRings()
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    )

    if num_aromatic_rings >= 2 and num_nitrogens >= 2:
        base = "quinazoline-like"
    elif num_aromatic_rings >= 2 and num_nitrogens == 1:
        base = "quinoline-like"
    elif num_aromatic_rings == 1 and num_nitrogens >= 1:
        base = "pyridine-like"
    elif num_aromatic_rings >= 1:
        base = "aromatic"
    else:
        base = "aliphatic"

    heteroatoms = []
    if num_nitrogens:
        heteroatoms.append(f"N{num_nitrogens}" if num_nitrogens > 1 else "N")
    if num_oxygens:
        heteroatoms.append(f"O{num_oxygens}" if num_oxygens > 1 else "O")

    halogens = []
    for symbol in ('F', 'Cl', 'Br', 'I'):
        count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == symbol)
        if count:
            halogens.append(f"{symbol}{count}" if count > 1 else symbol)

    formula = f"C{num_carbons}" + ("_" + "_".join(heteroatoms) if heteroatoms else "")
    if halogens:
        formula += "_" + "_".join(halogens)

    ring_part = f"{num_rings}ring{'s' if num_rings != 1 else ''}" if num_rings else "acyclic"

    return f"{base}_{formula}_{ring_part}"


def _pick_samples(df, n):
    active = df[df['Labels'] == 1]
    inactive = df[df['Labels'] == 0]
    n_active = min(n // 2 + n % 2, len(active))
    n_inactive = min(n - n_active, len(inactive))
    return active.sample(n_active) if n_active else active, inactive.sample(n_inactive) if n_inactive else inactive


def draw_sample_molecules(df, n=3, save_path=None):
    """Grid of n molecules, tries to keep it roughly balanced between
    active and inactive if both classes are available."""
    active_samples, inactive_samples = _pick_samples(df, n)
    samples = pd.concat([active_samples, inactive_samples]).head(n)

    mols, legends = [], []
    for _, row in samples.iterrows():
        mol = Chem.MolFromSmiles(row['Smiles'])
        if mol is None:
            continue
        mols.append(mol)
        label = structural_label(row['Smiles'])
        activity = "Active" if row['Labels'] == 1 else "Inactive"
        legends.append(f"{label}\n{activity}\nMW {row.get('MolWt', float('nan')):.1f}  LogP {row.get('LogP', float('nan')):.2f}")

    if not mols:
        print("no valid molecules to draw")
        return None

    img = Draw.MolsToGridImage(mols, molsPerRow=len(mols), subImgSize=(400, 400), legends=legends)
    display(img)
    if save_path:
        img.save(save_path)
    return img


def draw_extreme_molecules(df, save_path=None):
    """Grabs the highest MW, lowest MW, and highest LogP molecules —
    the edge cases in the descriptor distribution, worth a quick look
    to make sure nothing weird snuck into the dataset."""
    if 'MolWt' not in df.columns or 'LogP' not in df.columns:
        print("MolWt/LogP not in this dataframe, can't pick extremes")
        return None

    high_mw = df.nlargest(1, 'MolWt')
    low_mw = df.nsmallest(1, 'MolWt')
    high_logp = df.nlargest(1, 'LogP')

    cases = pd.concat([high_mw, low_mw, high_logp]).drop_duplicates()
    categories = {}
    for idx in high_mw.index:
        categories[idx] = "highest MW"
    for idx in low_mw.index:
        categories.setdefault(idx, "lowest MW")
    for idx in high_logp.index:
        categories.setdefault(idx, "highest LogP")

    mols, legends = [], []
    for idx, row in cases.iterrows():
        mol = Chem.MolFromSmiles(row['Smiles'])
        if mol is None:
            continue
        mols.append(mol)
        label = structural_label(row['Smiles'])
        activity = "Active" if row['Labels'] == 1 else "Inactive"
        legends.append(f"{label}\n{categories.get(idx, '')}\n{activity}\nMW {row['MolWt']:.1f}  LogP {row['LogP']:.2f}")

    if not mols:
        print("no valid molecules to draw")
        return None

    img = Draw.MolsToGridImage(mols, molsPerRow=len(mols), subImgSize=(400, 400), legends=legends)
    display(img)
    if save_path:
        img.save(save_path)
    return img


def describe_molecule(row):
    """Prints the descriptor breakdown for one row — just the numbers,
    doesn't draw anything. Pair it with Draw.MolToImage(mol) in a
    notebook if you want the structure alongside it."""
    label = structural_label(row['Smiles'])
    activity = "Active" if row['Labels'] == 1 else "Inactive"

    print(f"{label} ({activity})")
    print(f"  SMILES: {row['Smiles']}")
    for field in ('MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
                  'NumRotatableBonds', 'NumAromaticRings', 'HeavyAtomCount'):
        if field in row and not (isinstance(row[field], float) and pd.isna(row[field])):
            print(f"  {field}: {row[field]}")

    if all(f in row for f in ('MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors')):
        violations = []
        if row['MolWt'] > 500:
            violations.append("MW > 500")
        if row['LogP'] > 5:
            violations.append("LogP > 5")
        if row['NumHDonors'] > 5:
            violations.append("HBD > 5")
        if row['NumHAcceptors'] > 10:
            violations.append("HBA > 10")
        if len(violations) <= 1:
            note = "Lipinski compliant" if not violations else f"Lipinski compliant (1 violation: {violations[0]})"
        else:
            note = f"Lipinski non-compliant ({len(violations)} violations: {', '.join(violations)})"
        print(f"  {note}")


def view_individual_molecules(df, n=3):
    """Goes through molecules one at a time, structure plus full
    descriptor breakdown underneath each one — slower to read than a
    grid but more detail per molecule."""
    active_samples, inactive_samples = _pick_samples(df, n)
    samples = pd.concat([active_samples, inactive_samples]).head(n)

    for i, (_, row) in enumerate(samples.iterrows(), 1):
        mol = Chem.MolFromSmiles(row['Smiles'])
        if mol is None:
            print(f"molecule {i}: invalid SMILES, skipping")
            continue
        print(f"\n--- molecule {i}/{len(samples)} ---")
        img = Draw.MolToImage(mol, size=(400, 400))
        display(img)
        describe_molecule(row)