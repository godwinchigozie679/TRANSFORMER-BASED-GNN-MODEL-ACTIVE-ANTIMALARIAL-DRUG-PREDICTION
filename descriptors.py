"""
RDKit descriptor calculation. Kept as one flat list of (name, function)
pairs so adding/removing a descriptor is a one-line change.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

DESCRIPTOR_FUNCS = [
    ('MolWt', Descriptors.MolWt),
    ('LogP', Descriptors.MolLogP),
    ('NumHDonors', Descriptors.NumHDonors),
    ('NumHAcceptors', Descriptors.NumHAcceptors),
    ('TPSA', Descriptors.TPSA),
    ('NumRotatableBonds', Descriptors.NumRotatableBonds),
    ('NumAromaticRings', Descriptors.NumAromaticRings),
    ('NumAliphaticRings', Descriptors.NumAliphaticRings),
    ('NumSaturatedRings', Descriptors.NumSaturatedRings),
    ('NumHeteroatoms', Descriptors.NumHeteroatoms),
    ('RingCount', Descriptors.RingCount),
    ('HeavyAtomCount', Descriptors.HeavyAtomCount),
    ('MolMR', Descriptors.MolMR),
    ('BalabanJ', Descriptors.BalabanJ),
    ('BertzCT', Descriptors.BertzCT),
    ('Chi0', Descriptors.Chi0),
    ('Chi1', Descriptors.Chi1),
    ('Kappa1', Descriptors.Kappa1),
    ('Kappa2', Descriptors.Kappa2),
    ('LabuteASA', Descriptors.LabuteASA),
    ('LogS', Descriptors.EState_VSA1),  # rough solubility proxy, not a real logS
    ('SlogP_VSA1', Descriptors.SlogP_VSA1),
    ('SlogP_VSA2', Descriptors.SlogP_VSA2),
    ('SlogP_VSA3', Descriptors.SlogP_VSA3),
    ('SMR_VSA1', Descriptors.SMR_VSA1),
    ('SMR_VSA5', Descriptors.SMR_VSA5),
    ('SMR_VSA7', Descriptors.SMR_VSA7),
    ('PEOE_VSA1', Descriptors.PEOE_VSA1),
    ('PEOE_VSA6', Descriptors.PEOE_VSA6),
    ('PEOE_VSA8', Descriptors.PEOE_VSA8),
    ('PEOE_VSA10', Descriptors.PEOE_VSA10),
    ('PEOE_VSA14', Descriptors.PEOE_VSA14),
    ('NumSaturatedCarbocycles', Descriptors.NumSaturatedCarbocycles),
    ('NumSaturatedHeterocycles', Descriptors.NumSaturatedHeterocycles),
    ('NumAromaticCarbocycles', Descriptors.NumAromaticCarbocycles),
    ('NumAromaticHeterocycles', Descriptors.NumAromaticHeterocycles),
    ('FractionCsp3', Descriptors.FractionCSP3),
    ('HallKierAlpha', Descriptors.HallKierAlpha),
    ('Ipc', Descriptors.Ipc),
    ('Kappa3', Descriptors.Kappa3),
    ('Chi0v', Descriptors.Chi0v),
    ('Chi1v', Descriptors.Chi1v),
    ('Chi2v', Descriptors.Chi2v),
    ('Chi3v', Descriptors.Chi3v),
    ('Chi4v', Descriptors.Chi4v),
    ('Chi0n', Descriptors.Chi0n),
    ('Chi1n', Descriptors.Chi1n),
    ('Chi2n', Descriptors.Chi2n),
    ('Chi3n', Descriptors.Chi3n),
    ('Chi4n', Descriptors.Chi4n),
    ('MaxEStateIndex', Descriptors.MaxEStateIndex),
    ('MinEStateIndex', Descriptors.MinEStateIndex),
    ('MaxAbsEStateIndex', Descriptors.MaxAbsEStateIndex),
    ('MinAbsEStateIndex', Descriptors.MinAbsEStateIndex),
    ('MaxPartialCharge', Descriptors.MaxPartialCharge),
    ('MinPartialCharge', Descriptors.MinPartialCharge),
    ('MaxAbsPartialCharge', Descriptors.MaxAbsPartialCharge),
    ('MinAbsPartialCharge', Descriptors.MinAbsPartialCharge),
    ('EState_VSA2', Descriptors.EState_VSA2),
    ('EState_VSA3', Descriptors.EState_VSA3),
    ('EState_VSA4', Descriptors.EState_VSA4),
    ('EState_VSA5', Descriptors.EState_VSA5),
    ('EState_VSA6', Descriptors.EState_VSA6),
    ('EState_VSA7', Descriptors.EState_VSA7),
    ('EState_VSA8', Descriptors.EState_VSA8),
    ('EState_VSA9', Descriptors.EState_VSA9),
    ('EState_VSA10', Descriptors.EState_VSA10),
    ('EState_VSA11', Descriptors.EState_VSA11),
    ('VSA_EState1', Descriptors.VSA_EState1),
    ('VSA_EState2', Descriptors.VSA_EState2),
    ('VSA_EState3', Descriptors.VSA_EState3),
    ('VSA_EState4', Descriptors.VSA_EState4),
    ('VSA_EState5', Descriptors.VSA_EState5),
    ('VSA_EState6', Descriptors.VSA_EState6),
    ('VSA_EState7', Descriptors.VSA_EState7),
    ('VSA_EState8', Descriptors.VSA_EState8),
    ('VSA_EState9', Descriptors.VSA_EState9),
    ('VSA_EState10', Descriptors.VSA_EState10),
    ('fr_Al_COO', Descriptors.fr_Al_COO),
    ('fr_Al_OH', Descriptors.fr_Al_OH),
    ('fr_Ar_N', Descriptors.fr_Ar_N),
    ('fr_Ar_NH', Descriptors.fr_Ar_NH),
    ('fr_Ar_OH', Descriptors.fr_Ar_OH),
    ('fr_C_O', Descriptors.fr_C_O),
    ('fr_COO', Descriptors.fr_COO),
    ('fr_COO2', Descriptors.fr_COO2),
    ('fr_N_O', Descriptors.fr_N_O),
    ('fr_NH0', Descriptors.fr_NH0),
    ('fr_NH1', Descriptors.fr_NH1),
    ('fr_NH2', Descriptors.fr_NH2),
    ('fr_benzene', Descriptors.fr_benzene),
    ('fr_furan', Descriptors.fr_furan),
    ('fr_imidazole', Descriptors.fr_imidazole),
    ('fr_piperdine', Descriptors.fr_piperdine),
    ('fr_pyridine', Descriptors.fr_pyridine),
    ('fr_thiazole', Descriptors.fr_thiazole),
    ('fr_thiophene', Descriptors.fr_thiophene),
    ('NumBridgeheadAtoms', rdMolDescriptors.CalcNumBridgeheadAtoms),
    ('NumSpiroAtoms', rdMolDescriptors.CalcNumSpiroAtoms),
    ('NumAmideBonds', rdMolDescriptors.CalcNumAmideBonds),
    ('NumHBA', rdMolDescriptors.CalcNumHBA),
    ('NumHBD', rdMolDescriptors.CalcNumHBD),
    ('NumLipinskiHBA', rdMolDescriptors.CalcNumLipinskiHBA),
    ('NumLipinskiHBD', rdMolDescriptors.CalcNumLipinskiHBD),
]


def compute_descriptors(df, smiles_col='Smiles'):
    """Add one column per descriptor in DESCRIPTOR_FUNCS. Molecules that
    fail a given descriptor get NaN for that column rather than dropping
    the whole row — some descriptors (e.g. Ipc on large rings) occasionally
    blow up numerically, that's not a reason to lose the rest of the data."""
    values = {name: [np.nan] * len(df) for name, _ in DESCRIPTOR_FUNCS}
    failures = {}

    for row_pos, (idx, smiles) in enumerate(df[smiles_col].items()):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        for name, func in DESCRIPTOR_FUNCS:
            try:
                values[name][row_pos] = func(mol)
            except Exception:
                failures[name] = failures.get(name, 0) + 1

    descriptor_df = pd.DataFrame(values, index=df.index)
    out = pd.concat([df.copy(), descriptor_df], axis=1)

    if failures:
        print("Descriptor calculation failures (molecule still kept, just NaN for that column):")
        for name, count in failures.items():
            print(f"  {name}: {count}")

    return out
