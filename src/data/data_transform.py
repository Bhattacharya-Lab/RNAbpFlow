
#Partially from OpenFold
#Credit: https://github.com/aqlaboratory/openfold


import torch, os, sys
from src.data import base_constants as rc
from Bio import SeqIO
import numpy as np

def get_one_hot(fastaseq):
    
    nucleotide_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    
    try:
        indices = [nucleotide_map[n] for n in fastaseq.upper()]
    except KeyError:
        raise ValueError("Unknown nucleotide in sequence. Only 'A', 'U', 'G', 'C' are allowed.")

    onehot = np.eye(4)[indices]
    
    return onehot


def sequence_to_tensor(sequence):
    
    nucleotide_map = {'A': 0, 'G': 1, 'C': 2, 'U': 3, 'X': 4}
    mapped_sequence = [nucleotide_map[nt] for nt in sequence]
    tensor_idx = torch.tensor(mapped_sequence, dtype=torch.long)
    
    return tensor_idx

def genMap(res_idx):

    restype_atom28_to_atom23 = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom23_names[rc.restype_1to3[rt]]
        
        atom_name_to_idx23 = {name: i for i, name in enumerate(atom_names)}
        
        restype_atom28_to_atom23.append(
            [
                (atom_name_to_idx23[name] if name in atom_name_to_idx23 else 0)
                for name in rc.atom_types
            ]
        )

    restype_atom28_to_atom23.append([0] * 28)

    restype_atom28_to_atom23 = torch.tensor(
            restype_atom28_to_atom23,
            dtype=torch.int32,
        )

    residx_atom28_to_atom23 = restype_atom28_to_atom23[res_idx]

    return residx_atom28_to_atom23

def make_atom_mask(rna_target, input_dir):
    
    rnafasta = f"{input_dir}/{rna_target}/{rna_target}.fasta"

    if not os.path.exists(rnafasta):
        sys.exit(f"Couldn't locate the fasta file {rnafasta}, please provide a fasta file as input.")
    
    for record in SeqIO.parse(rnafasta, "fasta"):
        rnaseq = str(record.seq)
    
    res_idx = sequence_to_tensor(rnaseq)

    atom_map = genMap(res_idx)

    onehot = get_one_hot(rnaseq)
    
    return {
        "atom_map": atom_map,
        "aatype": res_idx,
        "onehot": onehot
    }
    
