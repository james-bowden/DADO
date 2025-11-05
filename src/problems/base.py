import numpy as np
import jax.numpy as jnp
from typing import Union, Optional, Tuple, Any, Callable, List
from abc import ABC, abstractmethod


class Objective(ABC):
    """Abstract base class for objective functions."""
    
    def __init__(self, D: List[int], *,
                 N: Optional[int] = None, 
                 negate: Optional[bool] = False, best_x: Optional[np.ndarray] = None,
                 weight_fn_name: Optional[str] = 'id', verbose: Optional[bool] = False):
        """
        Initialize an objective function.
        
        Args:
            D: Dimension of each position in the sequence
            L: Length of sequence
            N: Size of the landscape
            negate: Whether to negate the objective (True) or not (False). Always maximizes.
            best_x: Global optimum of the objective (if known). Always maximizes.
        """
        self.D = D
        self.L = 1 if isinstance(D, int) else len(D)
        self.N = N
        self.negate = negate
        self.best_x = best_x
        self.verbose = verbose
        self.weight_fn_name = weight_fn_name
        self.weight_fn = lambda x: x # NOTE: default to identity.
        self.apply_weight()
        # NOTE: can't query until after weight_fn is set.
        self.best_val = self.query(self.best_x[None, :])[0] if best_x is not None else None
        self.complete = False
        self.obj_name = "Objective"

    def apply_weight(self):
        match self.weight_fn_name:
            case 'id':
                pass
            case 'pos' | 'positive' | 'pos_scale' | 'exp_scale':
                raise NotImplementedError("Need minimum/maximum; haven't done for non-synthetic objectives yet.")
                # self.w_make_positive(baseline=np.min(self.values))
            case 'exp' | 'exponential':
                self.w_exponentiate()
            case 'fg_pos' | 'fg_pos_scale' | 'fg_exp' | 'fg_exp_scale':
                raise NotImplementedError("Cannot use fg transforms for non-graph objectives.")
            case _:
                raise ValueError(f"Invalid weight function: {self.weight_fn}")

    def w_make_positive(self, baseline: float = 0):
        """
        Make positive by setting a baseline.
        """
        baseline = jnp.minimum(baseline, 0)

        def apply_weight_fn(x: jnp.ndarray):
            return x - baseline
            
        self.weight_fn = apply_weight_fn
        # self.baseline = baseline

    def w_make_positive_scale(self, baseline: float):
        """
        Make positive by setting a baseline.
        """
        baseline = jnp.minimum(baseline, 0)
        max_value = jnp.max(self.values - baseline)
        
        def apply_weight_fn(x: jnp.ndarray):
            return (x - baseline) / max_value
            
        self.weight_fn = apply_weight_fn
        

    def w_exponentiate(self):
        """
        Exponentiate all node, edge functions.
        """
        def apply_weight_fn(x: jnp.ndarray):
            return jnp.exp(x)
            
        self.weight_fn = apply_weight_fn

    def w_exponentiate_scale(self):
        """
        Exponentiate all node, edge functions.
        """
        max_value = jnp.max(jnp.exp(self.values))
        def apply_weight_fn(x: jnp.ndarray):
            return jnp.exp(x) / max_value
        
        self.weight_fn = apply_weight_fn

    # def query(self, x: np.ndarray) -> np.ndarray:
    #     """
    #     Evaluate the objective function at the given points.
    #     """
    #     return np.array([self.query_single(x_i) for x_i in x])

    @abstractmethod
    def query(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the objective function at the given points.
        """
        pass

    # def query_jnp(self, x: jnp.ndarray) -> jnp.ndarray: # could override this to use vmap.
    #     """
    #     Evaluate the objective function at the given points.
    #     """
    #     return jnp.array(self.query(np.array(x)))

    # @abstractmethod # NOTE: only thing needed to implement.
    # # NOTE: make this give leading dim for query (vmapped)
    # def query_single(self, x: np.ndarray) -> float:
    #     """
    #     Evaluate the objective function at the given point.
    #     """
    #     pass

    # def query_single_jnp(self, x: jnp.ndarray) -> float: # could override this to use vmap.
    #     """
    #     Evaluate the objective function at the given point.
    #     """
    #     return self.query_single(np.array(x))
    
    # def to_onehot(self, x: np.ndarray) -> np.ndarray:
    #     """
    #     Convert categorical representation to one-hot representation.
        
    #     Args:
    #         x: Array of shape (m, L) with values in [0, D)
            
    #     Returns:
    #         Array of shape (m, L, D) with one-hot encoded values
    #     """
    #     # is_valid = self.is_valid(x)
    #     # if not np.all(is_valid):
    #     #     raise ValueError(f"Invalid input detected at indices: {np.where(~is_valid)[0]}")
        
    #     m = x.shape[0]
    #     result = jnp.zeros((m, self.L, self.D)).astype(jnp.bool_)
        
    #     # Use advanced indexing to set the appropriate elements to 1
    #     idx1 = jnp.arange(m)[:, None]
    #     idx2 = jnp.arange(self.L)[None, :]
    #     result = result.at[idx1, idx2, x.astype(int)].set(1)
        
    #     return result
    
    # def from_onehot(self, x_onehot: np.ndarray) -> np.ndarray:
    #     """
    #     Convert one-hot representation to categorical representation.
        
    #     Args:
    #         x_onehot: Array of shape (m, L, D) or (L, D) with one-hot encoded values
            
    #     Returns:
    #         Array of shape (m, L) or (L,) with categorical values
    #     """
    #     # Find the index of the non-zero element in the last dimension
    #     # TODO: use jnp.where instead?, since boolean array input
    #     return jnp.argmax(x_onehot, axis=-1)
    
    @abstractmethod
    def is_valid(self, x: np.ndarray) -> np.ndarray:
        """
        Check if inputs are valid.
        
        Args:
            x: Array of shape (m, L) with values to check
            
        Returns:
            Boolean array of shape (m,) indicating validity of each input
        """
        pass
    
    def best(self) -> Tuple[np.ndarray, float]:
        """
        Return the global optimum and its objective value.
        
        Returns:
            Tuple of (best_x, query(best_x))
        """
        if self.best_x is None:
            raise ValueError("Global optimum is not defined for this objective")
        
        return self.best_x, self.query(self.best_x[None, :])
