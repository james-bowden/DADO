from Bio.PDB import MMCIFParser
from Bio.Data.IUPACData import protein_letters_3to1
import numpy as np

def chain_to_sequence(chain):
    # Biopython's dict uses lowercase keys
    three_to_one = {k.upper(): v for k, v in protein_letters_3to1.items()}
    
    residues = [res for res in chain if res.get_id()[0] == ' ']  # standard AAs only
    seq = ""
    for res in residues:
        resname = res.get_resname().upper()
        seq += three_to_one.get(resname, "X")
    return residues, seq

def residue_contact_map(
        cif_filename, sequence, active_inds=None, 
        threshold=4.5, edgelist=False, 
        verbose=False):
    """
    Extracts a binary contact map from a PDB file.
    Idea + default threshold from https://www.pnas.org/doi/10.1073/pnas.2109649118
    "Analysis of Empirical Protein Fitness Functions" section.
    Notes:
        We expect sequence to be a contiguous substring of the full sequence in the CIF file.
        You can further sub-index this sequence afterward.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", cif_filename)
    model = next(structure.get_models()) # just use the first model
    chain = next(model.get_chains())  # assumes single chain

    residues, residue_str = chain_to_sequence(chain)
    L = len(residues)
    try:
        if active_inds is None:
            ind_start = residue_str.index(sequence) # raises error if not found
            residues = residues[ind_start:ind_start+len(sequence)]
            if verbose:
                print(f"File has length {L}. S has length {len(sequence)}. S begins at {ind_start}.")
        else:
            assert sequence == "".join([residue_str[i] for i in active_inds]), "Sequence does not match active indices."
            residues = [residues[i] for i in active_inds]
            if verbose:
                print(f"File has length {L}. S has length {len(sequence)}. S not necessarily contiguous.")
    except Exception as e:
        print(f"Matching sequence to CIF file failed: {e}")
        print(f"Sequence: {sequence}")
        print(f"Residue string: {residue_str}")
        print(f"Active indices: {active_inds}")
        exit(6)
    
    L = len(residues)
    assert L == len(sequence), "Sequence length does not match number of residues."
    contact_map = np.zeros((L, L), dtype=bool)

    for i in range(L):
        atoms_i = list(residues[i].get_atoms())
        for j in range(i + 1, L): # NOTE: skip self-contacts
            atoms_j = list(residues[j].get_atoms())
            if any(atom_i - atom_j <= threshold for atom_i in atoms_i for atom_j in atoms_j):
                contact_map[i, j] = contact_map[j, i] = True
    if edgelist:
        # only lower triangle, excl. diag
        return np.sort(
            np.argwhere(
                np.tril(contact_map, k=-1)
            ), axis=1,
        ).tolist(), residue_str
    else:
        return contact_map, residue_str

def extract_ca_distogram(cif_filename):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", cif_filename)
    model = next(structure.get_models())
    chain = next(model.get_chains())  # assumes single chain

    # Extract Cα atoms from standard residues only
    ca_atoms = [res["CA"] for res in chain if res.get_id()[0] == ' ' and "CA" in res]
    L = len(ca_atoms)
    
    distogram = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L): # NOTE: skip self-contacts
            dist = ca_atoms[i] - ca_atoms[j]
            distogram[i, j] = distogram[j, i] = dist
    return distogram

