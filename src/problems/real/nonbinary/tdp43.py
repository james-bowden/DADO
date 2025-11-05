import numpy as np
import jax
import jax.numpy as jnp
import os
import pandas as pd

from src.problems.real.base import sequence_to_index, alphabet_to_integer
from src.problems.real.nonbinary.base import NonBinaryObjective

# NOTE: TDP43 has 20 aas; the last state is a stop codon.
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY*'

class TDP43(NonBinaryObjective):
    """
    TDP43 dataset of 84 positions with 21 states each.
    From https://mavenn.readthedocs.io/en/latest/datasets/dataset_tdp43.html
    """
    
    def __init__(self, *args, **kwargs):
        filepath = f"{os.environ['DIR_DATA']}/mavenn/tdp43.npz"
        self.WT_full = 'MSEYIRVTEDENDEPIEIPSEDDGTVLLSTVTAQFPGACGLRYRNPVSQCMRGVRLVEGILHAPDAGWGNLVYVVNYPKDNKRKMDETDASSAVKVKRAVQKTSDLIVLGLPWKTTEQDLKEYFSTFGEVLMVQVKKDLKTGHSKGFGFVRFTEYETQVKVMSQRHMIDGRWCDCKLPNSKQSQDEPLRSRKVFVGRCTEDMTEDELREFFSQYGDVMDVFIPKPFRAFAFVTFADDQIAQSLCGEDLIIKGISVHISNAEPKHNSNRQLERSGRFGGNPGGFGNQGGFGNSRGGGAGLGNNQGSNMGGGMNFGAFSINPAMMAAAQAALQSSWGMMGMLASQQNQSGPSGNNQNQGNMQREPNQAFGSGNNSYSGSNSGAAIGWGSASNAGSGSGFNGGFGSSMDSKSSGWGM'
        self.WT = 'GNSRGGGAGLGNNQGSNMGGGMNFGAFSINPAMMAAAQAALQSSWGMMGMLASQQNQSGPSGNNQNQGNMQREPNQAFGSGNNS'
        self.active_inds = None # NOTE: can rewrite WT above in terms of active_inds
        self.str_to_int = lambda x: alphabet_to_integer(x, alphabet=ALPHABET)
        if os.path.exists(filepath):
            data = np.load(filepath)
            x = data["genotypes"]
            y = data["values"]
        else:
            print(f"File not found: {filepath}. Downloading from mavenn...")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            import mavenn
            df = mavenn.load_example_dataset('tdp43')
            x = self.str_to_int(df["x"].values)
            values = np.array(df["y"].values)
            y = (values - values.min()) / (values.max() - values.min())
            np.savez(filepath, genotypes=x, values=y)

        self.genotypes = jnp.array(x)
        values = jnp.array(y)
        assert values.min() >= 0 and values.max() <= 1, "Values should be normalized to [0, 1]"
        best_x = self.genotypes[values.argmax()]

        def query_fn(x):
            ind = sequence_to_index(x, self.genotypes)
            return jax.lax.select(
                ind == -1,
                -jnp.inf,
                values[ind]
            )

        # Initialize the parent class -- max=True (defaults)
        super().__init__(
            D=[21]*84,
            values=values,
            best_x=best_x,
            query_fn=jax.vmap(query_fn),
            complete=False,
            *args, **kwargs)
        assert self.genotypes.min() == 0 and self.genotypes.max() <= max(self.D), "Genotypes should be 0-indexed and have max(D) as the last state."
        self.obj_name = "TDP43"
        
        if self.verbose: print(f"{self.obj_name}: {len(self.D)} variables with {np.prod(self.D)} states total and {self.values.size} data.")
