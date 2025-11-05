import numpy as np
import jax
import jax.numpy as jnp
import os
import pandas as pd

from src.problems.real.base import sequence_to_index, alphabet_to_integer
from src.problems.real.nonbinary.base import NonBinaryObjective

# NOTE: YNZC has 20 aas.
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

class YNZC(NonBinaryObjective):
    """
    YNZC dataset of 39 positions with 20 states each.
    From Tsuboyama mega-scale dataset.
    """
    
    def __init__(self, *args, **kwargs):
        filepath = f"{os.environ['DIR_DATA']}/proteingym/processed/YNZC.npz"
        self.WT = 'MISNAKIARINELAAKAKAGVITEEEKAEQQKLRQEYLK'
        self.active_inds = None
        self.str_to_int = lambda x: alphabet_to_integer(x, alphabet=ALPHABET)
        if os.path.exists(filepath):
            data = np.load(filepath)
            x = data["genotypes"]
            y = data["values"]
        else:
            print(f"File not found: {filepath}. Extracting from proteingym...")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df = pd.read_csv(f"{os.environ['DIR_DATA']}/proteingym/pg_sub/YNZC_BACSU_Tsuboyama_2023_2JVD.csv")
            x = self.str_to_int(np.hstack((
                df["mutated_sequence"].values, np.array([self.WT])
            )))
            values = np.hstack((
                df["DMS_score"].values, np.array([0]) # add 0 for WT. still not 100% sure correct.
            ))
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
            D=[20]*39,
            values=values,
            best_x=best_x,
            query_fn=jax.vmap(query_fn),
            complete=False,
            *args, **kwargs)
        assert self.genotypes.min() == 0 and self.genotypes.max() <= max(self.D), "Genotypes should be 0-indexed and have max(D) as the last state."
        self.obj_name = "YNZC"

        if self.verbose: print(f"{self.obj_name}: {len(self.D)} variables with {np.prod(self.D)} states total and {self.values.size} data.")