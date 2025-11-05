import numpy as np
import jax.numpy as jnp
import random

from src.decomposition.fgm.tabular.base import TabularFunctionalTree
from src.problems.synthetic.graph.tree.base import SyntheticTreeObjective

class ChainObjective5(SyntheticTreeObjective):
    """
    Implementation of a specific tree objective with L=2, D=20.
    Uses random node functions and finds the global optimum through exhaustive search.
    """
    
    def __init__(self, *args, **kwargs):
        # Set parameters
        L = 2
        D1 = 20
        D2 = 1000
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create node functions as a dictionary mapping node indices to D-vectors
        # node_functions = [jnp.array(np.random.randn(D)) for _ in range(L)]
        node_functions = [
            jnp.arange(1, D1+1) / D1,
            jnp.zeros(D2), # no node contribution
        ]

        # Create edge functions
        edges = [
            (0, 1),
        ]
        ef = jnp.zeros((D1, D2))
        # make one of the worse states have potential for growth
        # ef = ef.at[5,500:550].set(jnp.arange(1, 50+1) * (2. / 50.)) # linear edge function
        ef = ef.at[10,:].set((jnp.arange(1, D2+1) * (4. / D2)) - 2.) # linear edge function
        edge_functions = {
            edges[0]: ef,
        }
        
        # Create a functional tree with L nodes
        fg = TabularFunctionalTree(
            n_nodes=L, n_states_list=[D1, D2], edges=edges, 
            node_functions=node_functions, edge_functions=edge_functions,
            root=0
        )
        
        # Initialize the parent class -- max=True, best_x=None (defaults)
        super().__init__(D=[D1, D2], fg=fg, *args, **kwargs)
        self.obj_name = "ChainObjective5"

        self.best_x = jnp.array([5, 999])
        self.best_val = self.query(self.best_x[None, :]).item()

        if self.verbose: print(f"{self.obj_name}: {L} variables with {D1, D2} states each")
