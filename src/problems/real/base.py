import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Tuple

from src.problems.base import Objective


class RealObjective(Objective):
    """Base class for real objective functions."""
    
    def __init__(self, values, query_fn, complete=False, *args, **kwargs):
        """
        Initialize a real objective function.
        
        Args:
            D: Dimension of each position in the sequence
            maximize: Whether to maximize (True) or minimize (False) the objective
            best_x: Global optimum of the objective (if known)
        """
        self.values = values
        self._query = query_fn
        self.complete = complete
        super().__init__(*args, **kwargs)
        self.N = np.prod(self.D)
        # NOTE: must set / fill below in child class
        # self.filepath = None
        # self.genotype_to_index = {}
        self.obj_name = "RealObjective"
    
    def is_valid(self, x: np.ndarray) -> bool:
        """
        Check if inputs are valid. For synthetic objectives, inputs are valid
        if all elements of x are within [0, D).
        
        Args:
            x: Array of shape (L,) with values to check
            
        Returns:
            Boolean indicating validity of input
        """
        # Check if x has the right shape and is binary.
        return x.shape == (self.L,) and np.all((0 <= x) & (x < self.D))

    def is_valid_raise(self, x: np.ndarray) -> bool:
        """
        Check if inputs are valid. For synthetic objectives, inputs are valid
        if all elements of x are within [0, D).
        """
        if not self.is_valid(x):
            raise ValueError(f"Invalid input: {x}, valid values are in [0, {self.D}).") 

    def apply_weight(self):
        match self.weight_fn_name:
            case 'pos'|'positive':
                self.w_make_positive(baseline=jnp.min(self.values))
            case 'pos_scale':
                self.w_make_positive_scale(baseline=jnp.min(self.values))
            case 'exp_scale':
                self.w_exponentiate_scale()
            case _:
                super().apply_weight()

    def query(self, x: jnp.ndarray) -> float:
        """
        Evaluate the objective function at the given points.
        
        Args:
            x: Array of shape (m, L) with values to evaluate
        """
        result = self._query(x) # NOTE: to be set in child class; should be vmapped.
        
        return self.weight_fn(result)
    
    def query_all(self):
        return self.values


######## UTILITY FUNCTIONS #########

ALPHABET_STANDARD = 'ACDEFGHIKLMNPQRSTVWY'

def alphabet_to_integer(genotypes: np.ndarray, alphabet=ALPHABET_STANDARD) -> np.ndarray:
    # Convert each string to a list of ints
    map = {aa: i for i, aa in enumerate(sorted(alphabet))}
    return np.array([
        [map[c] for c in seq] for seq in genotypes
    ], dtype=np.int32)

@jax.jit
def sequence_to_index(sequence: jnp.ndarray, all_sequences: jnp.ndarray) -> int:
    """
    Convert a sequence to an index.
    """
    f = jax.vmap(lambda x: jnp.all(x == sequence))
    return jnp.nonzero(f(all_sequences), size=1, fill_value=-1)[0][0] # tuple w/ one item of shape (1,)

@jax.jit
def hash_rows(arr, base=31):
    # Assumes arr has shape (N, L) and integer entries
    # Use Horner's method for rolling hash
    L = arr.shape[-1]
    powers = base ** jnp.arange(L)  # shape (L,)
    return jnp.dot(arr, powers)     # shape (N,)

@jax.jit
def is_valid_query(query_seq: jnp.ndarray, valid_hashes: jnp.ndarray) -> bool:
    query_hash = hash_rows(query_seq).squeeze()  # shape (1,) or scalar
    return jnp.any(valid_hashes == query_hash)