import os
import numpy as np
import dataclasses
import residue_constants


@dataclasses.dataclass(frozen=True)
class Protein:
  """
  Protein structure representation.
  """

  # Cartesian coordinates of atoms in angstroms. The atom types correspond to
  # residue_constants.atom_types, i.e. the first three are N, CA, CB.
  atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

  # Amino-acid type for each residue represented as an integer between 0 and
  # 20, where 20 is 'X'.
  aatype: np.ndarray  # [num_res]

  # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
  # is present and 0.0 if not. This should be used for loss masking.
  atom_mask: np.ndarray  # [num_res, num_atom_type]

  # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
  residue_index: np.ndarray  # [num_res]

  # 0-indexed number corresponding to the chain in the protein that this residue
  # belongs to.
  chain_index: np.ndarray  # [num_res]

  # B-factors, or temperature factors, of each residue (in sq. angstroms units),
  # representing the displacement of the residue from its ground truth mean
  # value.
  b_factors: np.ndarray  # [num_res, num_atom_type]

  def __post_init__(self):
    if len(np.unique(self.chain_index)) > residue_constants.PDB_MAX_CHAINS:
      raise ValueError(
          f'Cannot build an instance with more than {residue_constants.PDB_MAX_CHAINS} chains '
          'because these cannot be written to PDB format.')


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')


def to_pdb(prot: Protein, model=1, add_end=True) -> str:
  """Converts a `Protein` instance to a PDB string.

  Args:
    prot: The protein to convert to PDB.

  Returns:
    PDB string.
  """
  restypes = residue_constants.restypes + ['X']
  res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], 'UNK')
  atom_types = residue_constants.atom_types

  pdb_lines = []

  atom_mask = prot.atom_mask
  aatype = prot.aatype
  atom_positions = prot.atom_positions
  residue_index = prot.residue_index.astype(np.int32)
  chain_index = prot.chain_index.astype(np.int32)
  b_factors = prot.b_factors

  if np.any(aatype > residue_constants.restype_num):
    raise ValueError('Invalid aatypes.')

  # Construct a mapping from chain integer indices to chain ID strings.
  chain_ids = {}
  for i in np.unique(chain_index):  # np.unique gives sorted output.
    if i >= residue_constants.PDB_MAX_CHAINS:
      raise ValueError(
          f'The PDB format supports at most {residue_constants.PDB_MAX_CHAINS} chains.')
    chain_ids[i] = residue_constants.PDB_CHAIN_IDS[i]

  pdb_lines.append(f'MODEL     {model}')
  atom_index = 1
  last_chain_index = chain_index[0]
  # Add all atom sites.
  for i in range(aatype.shape[0]):
    # Close the previous chain if in a multichain PDB.
    if last_chain_index != chain_index[i]:
      pdb_lines.append(_chain_end(
          atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]],
          residue_index[i - 1]))
      last_chain_index = chain_index[i]
      atom_index += 1  # Atom index increases at the TER symbol.

    res_name_3 = res_1to3(aatype[i])
    for atom_name, pos, mask, b_factor in zip(
        atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
      if mask < 0.5:
        continue

      record_type = 'ATOM'
      name = atom_name if len(atom_name) == 4 else f' {atom_name}'
      alt_loc = ''
      insertion_code = ''
      occupancy = 1.00
      element = atom_name[0]  # Protein supports only C, N, O, S, this works.
      charge = ''
      # PDB is a columnar format, every space matters here!
      atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                   f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                   f'{residue_index[i]:>4}{insertion_code:>1}   '
                   f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                   f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                   f'{element:>2}{charge:>2}')
      pdb_lines.append(atom_line)
      atom_index += 1

  # Close the final chain.
  pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]),
                              chain_ids[chain_index[-1]], residue_index[-1]))
  pdb_lines.append('ENDMDL')
  if add_end:
    pdb_lines.append('END')

  # Pad all lines to 80 characters.
  pdb_lines = [line.ljust(80) for line in pdb_lines]
  return '\n'.join(pdb_lines) + '\n'  # Add terminating newline.


def create_bb_prot(model_pos, residues, chain):
    ca_idx = residue_constants.atom_order['CA']
    n = model_pos.shape[0]
    # if len(residues) < n:
    #     for _ in range(n-len(residues)):
    #         residues += "A"
    aatypes = np.array([residue_constants.restype_order[aa] for aa in residues], dtype=int)
    imputed_atom_pos = np.zeros([n, 37, 3])
    imputed_atom_pos[:, ca_idx] = model_pos
    imputed_atom_mask = np.zeros([n, 37])
    imputed_atom_mask[:, ca_idx] = 1.0
    residue_index = np.arange(n)
    chain_index = np.zeros(n) + residue_constants.chain2idx[chain]
    b_factors = np.zeros([n, 37])
    return Protein(
      atom_positions=imputed_atom_pos,
      atom_mask=imputed_atom_mask,
      aatype=aatypes,
      residue_index=residue_index,
      chain_index=chain_index,
      b_factors=b_factors)


def save_bb_as_pdb(bb_positions, residues, chain, fn):
    """save_bb_as_pdb saves generated c-alpha positions as a pdb file

    Args:
        bb_positions: c-alpha coordinates (before upscaling) of shape [seq_len, 3],
            not including masked residues

    """
    with open(fn, 'w') as f:
        # since trained on downscaled data, scale back up appropriately
        prot_pos = bb_positions
        bb_prot = create_bb_prot(prot_pos, residues, chain)
        pdb_prot = to_pdb(bb_prot, model=1, add_end=True)
        f.write(pdb_prot)