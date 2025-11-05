import numpy as np
import jax.numpy as jnp
import random

from src.decomposition.graphs import Graph
from src.decomposition.fgm.tabular.base import TabularFunctionalJunctionTree
from src.problems.synthetic.graph.base import SyntheticGraphObjective

class JunctionTreeObjective3(SyntheticGraphObjective):
    """
    Junction tree-based synthetic objective with D=4, L=11.

    Tree-like topology with a root clique (nodes 0,1,2) and other cliques connected
    directly to it. Each child clique overlaps with the root on one node.
    """
    
    def __init__(self, seed=42, *args, **kwargs):
        D = 4
        np.random.seed(seed)
        random.seed(seed)
        L = 11
        
        # Root clique
        root_clique = [0, 1, 2]
        edges = [
            (0,1),(0,2),(1,2)
        ]
        
        # Define child cliques (overlapping one node with root)
        child_cliques = [
            [2,3],      # overlaps on node 2
            [1,4,5],    # overlaps on node 1
            [0,6,7],    # overlaps on node 0
            [2,8,9],    # overlaps on node 2
            [1,10]      # overlaps on node 1
        ]
        
        # Add edges for child cliques
        for clique in child_cliques:
            if len(clique) == 2:
                edges.append((clique[0], clique[1]))
            elif len(clique) == 3:
                edges.extend([(clique[0], clique[1]), (clique[0], clique[2]), (clique[1], clique[2])])
        
        self.graph_original = Graph(L, edges)
        
        n_states_list = [D] * L
        fg = TabularFunctionalJunctionTree.fixed_graph_random(
            self.graph_original.to_junction_tree(),
            n_states_list, seed=seed
        )
        
        SyntheticGraphObjective._apply_epistasis(fg, D, weight_fn_name='fg_pos_scale')
        
        super().__init__(D=n_states_list, fg=fg, *args, **kwargs)
        self.obj_name = "JunctionTreeObjective3"
        
        self.genotypes = jnp.array([[(x >> i) & 1 for i in range(L)] for x in range(2**L)])
        self.values = fg(self.genotypes)
        
        if self.verbose:
            print(f"{self.obj_name}: {L} variables with {D} states each")
            print(f"Junction tree structure with {len(edges)} edges")
            print(f"Root clique: {root_clique}")
            print(f"Child cliques: {child_cliques}")
