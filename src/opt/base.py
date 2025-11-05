from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, Tuple, List
import jax
import jax.numpy as jnp
import numpy as np  # Adding missing import
from src.decomposition.fgm.tabular.base import TabularFunctionalGraph
from src.decomposition.fgm.mlp.base import MLPFunctionalGraph


# TODO: account for partial search spaces (not just D^L).
class Optimizer(ABC):
    """Base class for optimization methods over discrete spaces."""
    
    def __init__(self, 
                 n_states: Union[int, List[int], jnp.ndarray],
                 objective_fn: Union[Callable[[jnp.ndarray], float], TabularFunctionalGraph, MLPFunctionalGraph],
                 *args,
                 verbose: bool = False,
                 seed: Optional[int] = None,
                 **kwargs):
        """
        Initialize the optimizer.
        
        Args:
            n_variables: Number of variables in the search space
            n_states: Number of possible states for each variable
            objective_fn: Either a callable that evaluates a point in the search space,
                         or a FunctionalTree whose evaluate() method will be used
            seed: Random seed for reproducibility (default: None)
        """
        if isinstance(n_states, int):
            self.n_states = [n_states]
            self.n_variables = 1
            print(f"WARNING: n_states is an int, not a list. Assuming n_variables=1.")
        elif isinstance(n_states, list):
            self.n_states = n_states
            self.n_variables = len(n_states)
        elif isinstance(n_states, jnp.ndarray):
            self.n_states = list(n_states) # convert JAX array to list
            self.n_variables = len(n_states)
        else:
            raise ValueError(f"Invalid type for n_states: {type(n_states)}")

        self.seed = seed
        self.verbose = verbose
        if self.seed is None:
            self.seed = np.random.randint(0, jnp.iinfo(jnp.int32).max)
        assert isinstance(self.seed, int), f"Seed must be an int, not {type(self.seed)}"
        self.rng = jax.random.PRNGKey(self.seed)  # Use numpy to generate a random seed for jax PRNGKey

        self.objective = objective_fn # will have query(), query_single() methods and fg attribute
    
    @abstractmethod
    def precompute(self) -> None:
        """
        Perform any necessary pre-computation before optimization.
        For naive methods, this may do nothing.
        For EDAs, this would train the density model.
        """
        pass
    
    @abstractmethod
    def get_solutions(self, n_points: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate n_points solutions in the search space.
        
        Args:
            n_points: Number of solutions to generate
            
        Returns:
            Tuple containing:
                - Array of shape (n_points, n_variables) containing the generated solutions
                - Array of shape (n_points,) containing the objective values for each solution. 
                    If the objective is not ground truth, this is the model estimate.
        """
        pass 