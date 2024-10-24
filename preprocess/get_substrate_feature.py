import rdkit
from rdkit import Chem
import torch
from rdkit.Chem.rdchem import BondType, HybridizationType


def get_ligand_atom_features(rdmol):
    num_atoms = rdmol.GetNumAtoms()
    atomic_number = []
    aromatic = []
    # sp, sp2, sp3 = [], [], []
    hybrid = []
    degree = []
    for atom_idx in range(num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        HYBRID_TYPES = {t: i for i, t in enumerate(HybridizationType.names.values())}
        hybrid.append(HYBRID_TYPES[hybridization])
        # sp.append(1 if hybridization == HybridizationType.SP else 0)
        # sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        # sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
        degree.append(atom.GetDegree())
    node_type = torch.tensor(atomic_number, dtype=torch.long)

    row, col = [], []
    for bond in rdmol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
    row = torch.tensor(row, dtype=torch.long)
    col = torch.tensor(col, dtype=torch.long)
    hs = (node_type == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=num_atoms).numpy()
    # need to change ATOM_FEATS accordingly
    feat_mat = np.array([atomic_number, aromatic, degree, num_hs, hybrid], dtype=np.long).transpose()
    return feat_mat


mol = Chem.MolFromMolFile(substrate_path, sanitize=False)
features = get_ligand_atom_features(mol)
pos = mol.GetConformer().GetPositions()
# taking category 1.1.1 as an example
data_dict = {"1.1.1": {"train": {"seq": [], "coor": [], "motif": [], "pdb": [], "ec4": [], "substrate": [], "binding": [], "substrate_coor": [], "substrate_feat": []},
                       "valid": {"seq": [], "coor": [], "motif": [], "pdb": [], "ec4": [], "substrate": [], "binding": [], "substrate_coor": [], "substrate_feat": []}}}
data["1.1.1"]["train"]["substrate_coor"].append(pos.tolist())
data["1.1.1"]["train"]["substrate_feat"].append(features.tolist())