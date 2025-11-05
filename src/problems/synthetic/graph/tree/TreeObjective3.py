import numpy as np
import jax.numpy as jnp
import random

from src.decomposition.fgm.tabular.base import TabularFunctionalTree
from src.problems.synthetic.graph.tree.base import SyntheticTreeObjective

class TreeObjective3(SyntheticTreeObjective):
    """
    Implementation of a specific tree objective with L=9, D=20.
    Uses random node functions and finds the global optimum through exhaustive search.
    """
    
    def __init__(self, *args, **kwargs):
        # Set parameters
        L = 9
        D = 20
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create node functions as a dictionary mapping node indices to D-vectors
        node_functions = [jnp.array(np.random.randn(D)) for _ in range(L)]

        # Create edge functions
        edges = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (3, 7),
            (3, 8),
        ]
        edge_functions = {edge: jnp.array(np.random.randn(D, D)) for edge in edges}
        
        # Create a functional tree with L nodes
        fg = TabularFunctionalTree(
            n_nodes=L, n_states_list=[D] * L, edges=edges, 
            node_functions=node_functions, edge_functions=edge_functions,
            root=0
        )
        
        # Initialize the parent class -- max=True, best_x=None (defaults)
        super().__init__(D=[D]*L, fg=fg, *args, **kwargs)
        self.obj_name = "TreeObjective3"

        if self.verbose: print(f"{self.obj_name}: {L} variables with {D} states each")
