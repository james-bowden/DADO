import numpy as np
import jax.numpy as jnp
import random

from src.decomposition.graphs import Graph
from src.decomposition.fgm.tabular.base import TabularFunctionalJunctionTree
from src.problems.synthetic.graph.base import SyntheticGraphObjective

class JunctionTreeObjective2(SyntheticGraphObjective):
    """
    A junction tree-based synthetic objective with D=2, L=11.
    
    First 5 nodes (0-4) have the same structure as JunctionTreeObjective1.
    Next 6 nodes (5-10) form a similar structure with one connecting edge 
    to the first component (connecting node 4 to node 5).
    
    Uses sign epistasis and reciprocal sign epistasis on edge functions.
    """
    
    def __init__(self, seed=42, *args, **kwargs):
        # Force binary variables for tractability
        D = 4
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # 11 nodes total
        L = 11
        
        # Define edges for the junction tree structure
        # First component (nodes 0-4): same as JunctionTreeObjective1
        first_component_edges = [
            (0, 1), (1, 2), (0, 2), (2, 3), (1, 4), (2, 4), (1, 3),
        ]
        
        # Second component (nodes 5-10): similar structure
        second_component_edges = [
            (5, 6), (6, 7), (5, 7), (7, 8), (6, 9), (7, 9), (6, 8), (8, 10), (9, 10)
        ]
        
        # Connecting edge between the two components
        connecting_edge = [(4, 5)]
        
        # Combine all edges
        edges = first_component_edges + second_component_edges + connecting_edge
        
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
        self.obj_name = "JunctionTreeObjective2"

        self.genotypes = jnp.array([[(x >> i) & 1 for i in range(L)] for x in range(2**L)])
        self.values = fg(self.genotypes)

        if self.verbose:
            print(f"{self.obj_name}: {L} variables with {D} states each")
            print(f"Junction tree structure with {len(edges)} edges")
            print(f"First component: nodes 0-4 with {len(first_component_edges)} edges")
            print(f"Second component: nodes 5-10 with {len(second_component_edges)} edges")
            print(f"Connecting edge: {connecting_edge[0]}")