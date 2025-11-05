import numpy as np
import jax.numpy as jnp
import random

from src.decomposition.fgm.tabular.base import TabularFunctionalGraph
from src.problems.synthetic.graph.additive.base import SyntheticAdditiveObjective
from src.opt.search.exhaustive_search import ExhaustiveSearch

class AdditiveObjective2(SyntheticAdditiveObjective):
    """
    Implementation of a specific additive objective with L=4, D=20.
    Uses random node functions and finds the global optimum through exhaustive search.
    """
    
    def __init__(self, *args, **kwargs):
        # Set parameters
        L = 4
        D = 20
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create node functions as a dictionary mapping node indices to D-vectors
        # node_functions = jnp.array(np.random.randn(L, D))
        node_functions = [jnp.array(np.random.randn(D)) for _ in range(L)]

        # NOTE: no edges for this objective
        
        # Create a functional graph with L nodes and no edges
        fg = TabularFunctionalGraph(n_nodes=L, n_states_list=[D]*L, edges=[], node_functions=node_functions, edge_functions={})
        
        # Initialize the parent class -- max=True, best_x=None (defaults)
        super().__init__(D=[D]*L, fg=fg, *args, **kwargs)
        self.obj_name = "AdditiveObjective2"

        # give it best val by optimizing each dimension
        self.best_x = jnp.array([jnp.argmax(node_functions[i]) for i in range(L)])
        self.best_val = self.query(self.best_x[None, :]).item()

        if self.verbose: print(f"{self.obj_name}: {L} variables with {D} states each")
