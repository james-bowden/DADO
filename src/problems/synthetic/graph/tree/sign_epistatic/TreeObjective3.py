import numpy as np
import jax.numpy as jnp
import random

from src.decomposition.fgm.tabular.base import TabularFunctionalTree
from src.problems.synthetic.graph.tree.base import SyntheticTreeObjective

class SE_TreeObjective3(SyntheticTreeObjective):
    """
    Implementation of a specific tree objective with L=9, D=20.
    Introduces sign epistasis or reciprocal sign epistasis at each edge.
    """
    
    def __init__(self, *args, **kwargs):
        # Set parameters
        L = 9
        D = 20
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create node functions as a dictionary mapping node indices to D-vectors
        node_functions = [jnp.array(0.1 * np.random.randn(D)) for _ in range(L)]

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

        edge_functions = {}

        for edge in edges:
            i, j = edge
            ef = np.zeros((D, D))

            # Pick random states for sign epistasis at each node
            state_i = np.random.choice(D)
            state_i_prime = (state_i + 1) % D  # simple different state

            state_j = np.random.choice(D)
            state_j_prime = (state_j + 1) % D  # simple different state

            # Randomly choose effect size (from normal dist)
            effect_size = abs(np.random.normal(loc=0.0, scale=2.0))  # scale can be adjusted

            # Decide if reciprocal sign epistasis
            reciprocal = np.random.choice([True, False])

            # For sign epistasis on node i mutation
            ef[state_i, state_j] = 0.0  # baseline
            ef[state_i_prime, state_j] = effect_size

            # For sign epistasis on node j mutation
            if reciprocal:
                ef[state_i, state_j_prime] = 0.0  # baseline
                ef[state_i, state_j_prime] -= effect_size

            # Add small noise to other entries
            for ii in range(D):
                for jj in range(D):
                    if (ii, jj) not in [
                        (state_i, state_j),
                        (state_i_prime, state_j),
                        (state_i, state_j_prime)
                    ]:
                        ef[ii, jj] = 0.05 * np.random.randn()

            edge_functions[edge] = jnp.array(ef)
        
        # Create a functional tree with L nodes
        fg = TabularFunctionalTree(
            n_nodes=L, n_states_list=[D] * L, edges=edges, 
            node_functions=node_functions, edge_functions=edge_functions,
            root=0
        )
        
        # Initialize the parent class -- max=True, best_x=None (defaults)
        super().__init__(D=[D]*L, fg=fg, *args, **kwargs)
        self.obj_name = "SE_TreeObjective3"

        if self.verbose: print(f"{self.obj_name}: {L} variables with {D} states each")
