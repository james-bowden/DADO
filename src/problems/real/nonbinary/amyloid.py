import numpy as np
import jax
import jax.numpy as jnp
import os
import pandas as pd

from src.problems.real.base import sequence_to_index, alphabet_to_integer
from src.problems.real.nonbinary.base import NonBinaryObjective

# NOTE: Amyloid has 20 aas; the last state is a stop codon.
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY*'

class Amyloid(NonBinaryObjective):
    """
    Amyloid dataset of 42 positions with 20 states each.
    From https://mavenn.readthedocs.io/en/latest/datasets/dataset_amyloid.html
    """
    
    def __init__(self, *args, **kwargs):
        filepath = f"{os.environ['DIR_DATA']}/mavenn/amyloid.npz"
        self.WT = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'
        self.active_inds = None
        self.str_to_int = lambda x: alphabet_to_integer(x, alphabet=ALPHABET)
        if os.path.exists(filepath):
            data = np.load(filepath)
            x = data["genotypes"]
            y = data["values"]
        else:
            print(f"File not found: {filepath}. Downloading from mavenn...")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            import mavenn
            df = mavenn.load_example_dataset('amyloid')
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
            D=[21]*42,
            values=values,
            best_x=best_x,
            query_fn=jax.vmap(query_fn),
            complete=False,
            *args, **kwargs)
        assert self.genotypes.min() == 0 and self.genotypes.max() <= max(self.D), "Genotypes should be 0-indexed and have max(D) as the last state."
        self.obj_name = "Amyloid"

        if self.verbose: print(f"{self.obj_name}: {len(self.D)} variables with {np.prod(self.D)} states total and {self.values.size} data.")