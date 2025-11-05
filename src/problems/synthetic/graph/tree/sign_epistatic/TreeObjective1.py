import numpy as np
import jax.numpy as jnp
import random

from src.decomposition.fgm.tabular.base import TabularFunctionalTree
from src.problems.synthetic.graph.tree.base import SyntheticTreeObjective

class SE_TreeObjective1(SyntheticTreeObjective):
    """
    Implementation of a specific tree objective with L=2, D=20.
    Uses random node functions and finds the global optimum through exhaustive search.
    """
    
    def __init__(self, *args, **kwargs):
        # Set parameters
        L = 2
        D = 20
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create node functions as a dictionary mapping node indices to D-vectors
        node_functions = [jnp.array(0.01 * np.random.randn(D)) for _ in range(L)]

        # Create edge functions
        edges = [
            (0, 1),
        ]
        # Edge functions: initialize as zeros
        ef = np.zeros((D, D))

        # Construct sign epistasis:
        # We'll use states 0 and 1 for simplicity
        # Node 1 in state 0:
        ef[0, 0] = 0.0
        ef[1, 0] = 1.0  # mutation at node 0 is beneficial

        # Node 1 in state 1:
        ef[0, 1] = 0.0
        ef[1, 1] = -1.0  # mutation at node 0 is deleterious

        # Fill other states with small random noise to avoid confounding effects
        for i in range(D):
            for j in range(D):
                if (i, j) not in [(0,0), (1,0), (0,1), (1,1)]:
                    ef[i,j] = 0.05 * np.random.randn()
        
        edge_functions = {
            edges[0]: jnp.array(ef),
        }

        edge_functions = {edge: jnp.array(np.random.randn(D, D)) for edge in edges}
        
        # Create a functional tree with L nodes
        fg = TabularFunctionalTree(
            n_nodes=L, n_states_list=[D] * L, edges=edges, 
            node_functions=node_functions, edge_functions=edge_functions,
            root=0
        )
        
        # Initialize the parent class -- max=True, best_x=None (defaults)
        super().__init__(D=[D]*L, fg=fg, *args, **kwargs)
        self.obj_name = "SE_TreeObjective1"

        if self.verbose: print(f"{self.obj_name}: {L} variables with {D} states each")
