import numpy as np
import jax.numpy as jnp
import random

from src.decomposition.graphs import GeneralizedGraph, Graph
from src.decomposition.fgm.tabular.base import TabularGeneralizedFunctionalGraph
from src.problems.synthetic.graph.base import SyntheticGraphObjective

class ComponentObjective1(SyntheticGraphObjective):
    """
    A test objective with multiple disconnected components:
    - 2 completely disconnected nodes
    - 2 separate random trees  
    - 2 random graphs that aren't trees
    
    Uses sign epistasis and reciprocal sign epistasis on edge functions.
    """
    
    def __init__(self, seed=42, *args, **kwargs):
        # Force binary variables for tractability with WHT
        D = 2
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Total number of nodes: 2 isolated + 3*2 + 4*2 = 16 nodes
        # Component structure:
        # - Nodes 0: isolated
        # - Nodes 2,3,4: tree 1 (chain)
        # - Nodes 1, 5,6,7: tree 2  
        # - Nodes 8,9,10,11, 15: graph 1 JT
        # - Nodes 12,13,14: graph 2 (complete graph on 3 nodes)
        
        L = 16
        
        # Define all edges for the generalized graph
        edges = [
            # Tree 1: 2-3-4 (chain)
            (2, 3), (3, 4),
            # Tree 2: 5-6-{7, 1}) 
            (5, 6), (6, 7), (6, 1),
            # Graph 1: 4-cycle (8-9-10)-{11, 15}
            # (8, 9), (9, 10), (8, 10), (10, 11), (9, 15), (10, 15),
            (8, 9), (9, 10), (8, 10), (10, 11), (9, 15), (10, 15), (9, 11),
            # Graph 2: Complete graph on nodes 12,13,14
            (12, 13), (12, 14), (13, 14),
        ]
        self.graph_original = Graph(L, edges)
        
        # Create the generalized functional graph
        n_states_list = [D] * L
        generalized_graph = GeneralizedGraph(L, edges, verbose=False)
        fg = TabularGeneralizedFunctionalGraph.fixed_graph_random(
            generalized_graph, n_states_list, seed=seed
        )
        
        # Apply sign epistasis and reciprocal sign epistasis to edge functions
        # We'll modify the edge functions in the trees and graphs
        SyntheticGraphObjective._apply_epistasis(fg, D, weight_fn_name='fg_pos_scale')
        
        # Initialize the parent class
        super().__init__(D=D, fg=fg, *args, **kwargs)

        self.genotypes = jnp.array([[(x >> i) & 1 for i in range(L)] for x in range(2**L)])
        self.values = fg(self.genotypes)
        self.obj_name = "ComponentObjective1"

        if self.verbose:
            print(f"{self.obj_name}: {L} variables with {D} states each")
            print(f"Components: 2 isolated nodes, 2 trees (3 nodes each), 2 graphs (4 nodes each)")
