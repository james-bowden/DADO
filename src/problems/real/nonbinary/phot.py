import numpy as np
import jax
import jax.numpy as jnp
import os
import pandas as pd
import ast

from src.problems.real.base import sequence_to_index, alphabet_to_integer
from src.problems.real.nonbinary.base import NonBinaryObjective

# NOTE: this file is also available in src/problems/real/pg_sub_stats_scrubbed.csv.
METADATA_PATH = f"{os.environ['DIR_DATA']}/proteingym/pg_sub/pg_sub_stats_scrubbed.csv"

class PHOT(NonBinaryObjective):
    """
    PHOT dataset of 118 mutagenized positions, with 20 states each.
    Contains as many as 15 mutations from the wild-type, meaning mutations are generally sparse.
    Total dataset size is 167530.
    From proteingym / https://pubs.acs.org/doi/10.1021/acssynbio.2c00662
    """
    
    def __init__(self, *args, **kwargs):
        filepath = f"{os.environ['DIR_DATA']}/proteingym/processed/phot.npz"
        self.WT_full = 'AGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA'
        extract_inds = lambda s: s
        self.WT = extract_inds(self.WT_full)
        self.active_inds = None
        self.str_to_int = lambda x: alphabet_to_integer(x)
        if os.path.exists(filepath):
            data = np.load(filepath)
            x = data["genotypes"]
            y = data["values"]
        else:
            print(f"File not found: {filepath}. Processing from proteingym...")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # TODO: have this directly download proteingym?
            csv_path = f"{os.environ['DIR_DATA']}/proteingym/pg_sub/PHOT_CHLRE_Chen_2023.csv"
            df = pd.read_csv(csv_path)
            x = self.str_to_int(
                    np.array([self.WT] + [extract_inds(s) for s in df['mutated_sequence']])
                )
            values = np.hstack(([0.0], df["DMS_score"].values))
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
            D=[20]*118,
            values=values,
            best_x=best_x,
            query_fn=jax.vmap(query_fn),
            complete=False,
            *args, **kwargs)
        assert self.genotypes.min() == 0 and self.genotypes.max() <= max(self.D), "Genotypes should be 0-indexed and have max(D) as the last state."
        self.obj_name = "PHOT"
        
        if self.verbose: print(f"{self.obj_name}: {len(self.D)} variables with {np.prod(self.D)} states total and {self.values.size} data.")
