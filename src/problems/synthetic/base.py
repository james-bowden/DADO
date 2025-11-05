import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Tuple

from src.problems.base import Objective
from src.opt.search.exhaustive_search import ExhaustiveSearch


class SyntheticObjective(Objective):
    """Base class for synthetic objective functions."""
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a synthetic objective function.
        
        Args:
            D: Dimension of each position in the sequence
            maximize: Whether to maximize (True) or minimize (False) the objective
            best_x: Global optimum of the objective (if known)
        """
        self.values = None
        super().__init__(*args, **kwargs)
        self.N = np.prod(self.D)
        self.complete = True # NOTE: need not be true, but all implemented so far are complete.
        self.obj_name = "SyntheticObjective"
    
    def is_valid(self, x: np.ndarray) -> bool:
        """
        Check if inputs are valid. For synthetic objectives, inputs are valid
        if all elements of x are within [0, D).
        
        Args:
            x: Array of shape (L,) with values to check
            
        Returns:
            Boolean indicating validity of input
        """
        # Check if x has the right shape
        return x.shape == (self.L,) and np.all((0 <= x) & (x < self.D))
        # if x.ndim != 2 or x.shape[1] != self.L:
        #     m = x.shape[0] if x.ndim > 0 else 1
        #     print(f"Invalid shape: {x.shape}")
        #     return np.zeros(m, dtype=bool)
        
        # # Check if all elements are in the range [0, D)
        # return np.all((0 <= x) & (x < self.D), axis=1)

    def is_valid_raise(self, x: np.ndarray) -> bool:
        """
        Check if inputs are valid. For synthetic objectives, inputs are valid
        if all elements of x are within [0, D).
        """
        if not self.is_valid(x):
            raise ValueError(f"Invalid input: {x}, valid values are in [0, {self.D}).") 

    def exhaustive_search(self, verbose: bool = False, **kwargs) -> Tuple[np.ndarray, float]:
        """
        Use exhaustive search to find the best solution and set the best_x and values attributes.
        """
        if self.values is not None and self.best_x is not None and self.best_val is not None:
            print("Warning: Overwriting existing exhaustive search results. OK if using pos weight fn.")
        
        opt = ExhaustiveSearch(n_variables=self.L, n_states=self.D, objective_fn=self, verbose=verbose)
        opt.precompute(**kwargs)
        best_x, best_val = opt.get_best_solution()

        self.values = opt.values
        self.best_x = best_x
        self.best_val = best_val

    def apply_weight(self):
        match self.weight_fn_name:
            case 'pos' | 'positive' | 'pos_scale' | 'exp_scale':
                print(f"Warning: Weight function {self.weight_fn_name} requires exhaustive search to find min/max. Commencing...")
                self.exhaustive_search()
                match self.weight_fn_name:
                    case 'pos' | 'positive':
                        self.w_make_positive(baseline=jnp.min(self.values))
                    case 'pos_scale':
                        self.w_make_positive_scale(baseline=jnp.min(self.values))
                    case 'exp_scale':
                        self.w_exponentiate_scale()
            case _:
                super().apply_weight()
