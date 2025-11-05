import numpy as np
import jax.numpy as jnp
import random

from src.decomposition.fgm.tabular.base import TabularGeneralizedFunctionalGraph
from src.decomposition.graphs import GeneralizedGraph
from src.problems.synthetic.graph.base import SyntheticGraphObjective
from src.opt.search.exhaustive_search import ExhaustiveSearch

class IndependentCliqueObjective2(SyntheticGraphObjective):
    """
    Implementation of independent clique objective with k=4 cliques of size=4, D=20.
    Each clique is fully connected internally but independent from other cliques.
    """
    
    def __init__(self, *args, **kwargs):
        # Set parameters
        k = 4  # number of cliques
        clique_size = 4  # size of each clique
        D = 20  # number of states per variable
        total_vars = k * clique_size  # total variables = 16
        
        # Set random seed for reproducibility
        np.random.seed(902)
        
        # Create edges for k independent cliques
        edges = []
        for clique_idx in range(k):
            start_node = clique_idx * clique_size
            clique_nodes = list(range(start_node, start_node + clique_size))
            
            # Add all edges within this clique (fully connected)
            for i in range(len(clique_nodes)):
                for j in range(i + 1, len(clique_nodes)):
                    edges.append((clique_nodes[i], clique_nodes[j]))
        
        # Create a generalized graph with these edges
        gg = GeneralizedGraph(total_vars, edges, verbose=False)
        
        # Create a functional graph from the generalized graph
        n_states_list = [D] * total_vars
        fg = TabularGeneralizedFunctionalGraph.fixed_graph_random(
            gg, n_states_list, seed=902
        )
        
        # Initialize the parent class
        super().__init__(D=[D]*total_vars, fg=fg, *args, **kwargs)
        self.obj_name = "IndependentCliqueObjective2"

        if self.verbose: 
            print(f"{self.obj_name}: {k} cliques of size {clique_size}, {total_vars} variables with {D} states each")