import numpy as np
import jax.numpy as jnp
import random

from src.decomposition.fgm.tabular.base import TabularFunctionalTree
from src.problems.synthetic.graph.tree.base import SyntheticTreeObjective

class ChainObjective2(SyntheticTreeObjective):
    """
    Implementation of a specific tree objective with L=2, D=20.
    Uses random node functions and finds the global optimum through exhaustive search.
    """
    
    def __init__(self, *args, **kwargs):
        # Set parameters
        L = 2
        D = 100
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create node functions as a dictionary mapping node indices to D-vectors
        # node_functions = [jnp.array(np.random.randn(D)) for _ in range(L)]
        node_functions = [
            jnp.arange(1, D+1) / 100.,
            jnp.zeros(D), # no node contribution
        ]

        # Create edge functions
        edges = [ # could remove this? idk if matters.
            (0, 1),
        ]
        ef = jnp.zeros((D, D))
        # make one of the worse states have potential for growth
        # ef = ef.at[10,50:].set(jnp.arange(50) * (2. / 50.)) # linear edge function
        edge_functions = {
            edges[0]: ef,
        }
        
        # Create a functional tree with L nodes
        fg = TabularFunctionalTree(
            n_nodes=L, n_states_list=[D] * L, edges=edges, 
            node_functions=node_functions, edge_functions=edge_functions,
            root=0
        )
        
        # Initialize the parent class -- max=True, best_x=None (defaults)
        super().__init__(D=[D]*L, fg=fg, *args, **kwargs)
        self.obj_name = "ChainObjective2"

        if self.verbose: print(f"{self.obj_name}: {L} variables with {D} states each")
