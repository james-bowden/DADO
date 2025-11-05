import numpy as np
import jax.numpy as jnp
import random

from src.decomposition.graphs import Graph
from src.decomposition.fgm.tabular.base import TabularFunctionalJunctionTree
from src.problems.synthetic.graph.base import SyntheticGraphObjective

class JunctionTreeObjective1(SyntheticGraphObjective):
    """
    A junction tree-based synthetic objective with D=2, L=5.
    Uses the same graph structure as the 5-node component from ComponentObjective1
    (nodes 8,9,10,11,15 with edges forming a junction tree).
    
    Uses sign epistasis and reciprocal sign epistasis on edge functions.
    """
    
    def __init__(self, seed=42, *args, **kwargs):
        # Force binary variables for tractability
        D = 20
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # 5 nodes total
        L = 5
        
        # Define edges for the junction tree structure
        # Based on ComponentObjective1's graph 1: nodes 8,9,10,11,15
        # Remapped to nodes 0,1,2,3,4 for L=5
        edges = [
            (0, 1), (1, 2), (0, 2), (2, 3), (1, 4), (2, 4), (1, 3),
        ]
        self.graph_original = Graph(L, edges)
        
        # Create the functional junction tree
        n_states_list = [D] * L
        fg = TabularFunctionalJunctionTree.fixed_graph_random(
            self.graph_original.to_junction_tree(), 
            n_states_list, seed=seed
        )
        
        # Apply sign epistasis and reciprocal sign epistasis to edge functions
        SyntheticGraphObjective._apply_epistasis(fg, D, weight_fn_name='fg_pos_scale')
        
        # Initialize the parent class
        super().__init__(D=n_states_list, fg=fg, *args, **kwargs)

        self.genotypes = jnp.array([[(x >> i) & 1 for i in range(L)] for x in range(2**L)])
        self.values = fg(self.genotypes)
        self.obj_name = "JunctionTreeObjective1"

        if self.verbose:
            print(f"{self.obj_name}: {L} variables with {D} states each")
            print(f"Junction tree structure with {len(edges)} edges")