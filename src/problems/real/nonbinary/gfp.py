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

class GFP(NonBinaryObjective):
    """
    GFP dataset of 233 mutagenized positions of total 238, with 20 states each.
    From proteingym / https://www.nature.com/articles/nature17995.
    UniProt accession: https://www.uniprot.org/uniprotkb/P42212/entry
    """
    
    def __init__(self, *args, **kwargs):
        filepath = f"{os.environ['DIR_DATA']}/proteingym/processed/gfp.npz"
        self.WT_full = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
        metadata = pd.read_csv(METADATA_PATH)
        # self.active_inds = ast.literal_eval(metadata[metadata['name'].str.contains('GFP_AEQVI')]['positions'].values[0])
        # extract_inds = lambda s: ''.join([s[i] for i in self.active_inds])
        self.active_inds = None
        extract_inds = lambda s: s
        self.WT = extract_inds(self.WT_full)
        # assert len(self.WT) == 233, "WT should be 233 residues long."
        self.str_to_int = lambda x: alphabet_to_integer(x)
        if os.path.exists(filepath):
            data = np.load(filepath)
            x = data["genotypes"]
            y = data["values"]
        else:
            print(f"File not found: {filepath}. Processing from proteingym...")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # TODO: have this directly download proteingym?
            csv_path = f"{os.environ['DIR_DATA']}/proteingym/pg_sub/GFP_AEQVI_Sarkisyan_2016.csv"
            df = pd.read_csv(csv_path)
            x = self.str_to_int(
                np.array([self.WT] + [extract_inds(s) for s in df['mutated_sequence']])
            )
            values = np.hstack(([0.0], df["DMS_score"].values))
            assert len(x) == len(values), "x and values should have the same length."
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
            D=[20]*len(self.WT),
            values=values,
            best_x=best_x,
            query_fn=jax.vmap(query_fn),
            complete=False,
            *args, **kwargs)
        assert self.genotypes.min() == 0 and self.genotypes.max() <= max(self.D), "Genotypes should be 0-indexed and have max(D) as the last state."
        self.obj_name = "GFP"
        
        if self.verbose: print(f"{self.obj_name}: {len(self.D)} variables with {np.prod(self.D)} states total and {self.values.size} data.")
