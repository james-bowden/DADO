import numpy as np
import jax.numpy as jnp
import random

from src.decomposition.fgm.tabular.base import TabularFunctionalTree
from src.problems.synthetic.graph.tree.base import SyntheticTreeObjective

class TreeObjective4(SyntheticTreeObjective):
    """
    Implementation of a specific tree objective with L=25, D=20.
    Uses random node functions and finds the global optimum through exhaustive search.
    """
    
    def __init__(self, *args, **kwargs):
        # Set parameters
        L = 25
        D = 20
        
        # Set random seed for reproducibility
        np.random.seed(48)
        
        # Create random tree
        ft = TabularFunctionalTree.random(
            n_nodes=L, n_states_list=[D] * L, seed=48
        )
        
        # Initialize the parent class -- max=True, best_x=None (defaults)
        super().__init__(D=[D]*L, fg=ft, *args, **kwargs)
        self.obj_name = "TreeObjective4"

        if self.verbose: print(f"{self.obj_name}: {L} variables with {D} states each")
