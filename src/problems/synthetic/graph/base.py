import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional, Union

from src.decomposition.graphs import Graph, JunctionTree, GeneralizedGraph
from src.decomposition.fgm.tabular.base import TabularFunctionalGraph
from src.decomposition.fgm.mlp.base import MLPFunctionalGraph, MLPGeneralizedFunctionalGraph
from src.problems.synthetic.base import SyntheticObjective


class SyntheticGraphObjective(SyntheticObjective):
    """
    Synthetic objective function based on a functional graph.
    The function value is computed by evaluating a functional graph.
    """
    
    def __init__(self, fg: Union[TabularFunctionalGraph, MLPFunctionalGraph], *args, **kwargs):
        """
        Initialize a synthetic graph objective function.
        
        Args:
            D: Dimension of each position in the sequence
            L: Length of sequence
            fg: Functional graph that defines the objective
            maximize: Whether to maximize (True) or minimize (False) the objective
            best_x: Global optimum of the objective (if known)
        """
        self.fg = fg # NOTE: must go first, b/c super will use in apply_weight.
        super().__init__(*args, **kwargs)
        if isinstance(fg, JunctionTree):
            # For junction trees, check that the number of original variables equals L
            assert len(self.D) == len(fg.original_n_states_list), f"Length of sequence (L={len(self.D)}) must equal number of original variables in junction tree ({len(fg.original_n_states_list)})."
        elif isinstance(fg, GeneralizedGraph):
            # For generalized graphs, check that total variables across all subgraphs equals L
            total_vars = sum(len(node_indices) for node_indices in fg.subgraph_node_indices)
            assert len(self.D) == total_vars, f"Length of sequence (L={len(self.D)}) must equal total variables in generalized graph ({total_vars})."
        else:
            # For regular functional graphs, check that L equals number of nodes
            assert len(self.D) == fg.n_nodes, f"Length of sequence (L={len(self.D)}) must equal number of nodes in functional graph ({fg.n_nodes})."
        self.obj_name = f"SyntheticGraphObjective"

    def apply_weight(self):
        match self.weight_fn_name:
            case 'fg_pos' | 'fg_exp' | 'fg_pos_scale' | 'fg_exp_scale':
                self.fg.apply_weight_fn(self.weight_fn_name)
            case _:
                super().apply_weight()
    
    @partial(jax.jit, static_argnums=(0,))
    def query(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the objective function at the given points using the functional graph.
        
        Args:
            x: Array of shape (m, L) with values to evaluate
        """
        if isinstance(self.fg, (MLPFunctionalGraph, MLPGeneralizedFunctionalGraph)):
            # print(f"Using MLPFGM split_merge")
            result = MLPFunctionalGraph._split_merge(self.fg).evaluate_batched(x)
        else: # tabular may have tracer errors and require similar handling, haven't tested.
            # print(f"Using self.fg(x)")
            result = self.fg(x)
        match self.negate: # TODO: don't know how this interacts with the weight functions.
            case True:
                return -self.weight_fn(jax.lax.stop_gradient(result))
            case False:
                return self.weight_fn(jax.lax.stop_gradient(result))

    @staticmethod
    def _apply_epistasis(fg, D, weight_fn_name='fg_pos_scale'):
        """Apply sign epistasis and reciprocal sign epistasis to edge functions.
        
        Args:
            fg: Either a FunctionalGraph or GeneralizedFunctionalGraph
            D: Number of states per variable
        """
        assert D >= 2, "D must be at least 2, o/w nothing to optimize."
        
        # Handle both GeneralizedFunctionalGraph and single functional graphs
        if hasattr(fg, 'functional_subgraphs'):
            # GeneralizedFunctionalGraph case
            functional_subgraphs = fg.functional_subgraphs
            subgraph_node_indices = fg.subgraph_node_indices
            # subgraphs = fg.subgraphs
        else:
            # Single functional graph case - wrap in a list
            functional_subgraphs = [fg]
            subgraph_node_indices = [list(range(fg.n_nodes))]
            # subgraphs = [fg]  # Assume single graph has tree-like structure if needed
        
        # Get the functional subgraphs (not isolated nodes)
        multi_node_subgraphs = []
        
        for i, subgraph in enumerate(functional_subgraphs):
            if len(subgraph_node_indices[i]) > 1:
                multi_node_subgraphs.append(subgraph)
        
        # Apply epistasis to all multi-node subgraphs
        for subgraph in multi_node_subgraphs:
            # edges_list = list(subgraph.edge_functions.keys())
            
            # Randomly choose epistasis type for this subgraph (50/50 split)
            use_reciprocal = np.random.random() < 0.5
            
            # Single loop over all edges
            for i, (edge, edge_func) in enumerate(subgraph.edge_functions.items()):
                new_func = jnp.zeros_like(edge_func)
                
                # TODO: make effect sizes a (random) parameter
                # TODO: make the second path epistatic too. (currently one path is always neg/pos epistatic)
                if not use_reciprocal or i % 2 == 0: # SE for even edges
                    # Sign epistasis: beneficial in one context, deleterious in another
                    new_func = new_func.at[0, 0].set(0.0)   # baseline
                    new_func = new_func.at[1, 0].set(1.5 if use_reciprocal else 1.0)   # beneficial
                    new_func = new_func.at[0, 1].set(0.0)   # baseline  
                    new_func = new_func.at[1, 1].set(-0.5 if use_reciprocal else -1.0)  # deleterious
                else:
                    # Reciprocal sign epistasis for odd edges: deleterious when partner is 0, beneficial when partner is 1
                    new_func = new_func.at[0, 0].set(0.0)
                    new_func = new_func.at[1, 0].set(-0.5)  # deleterious
                    new_func = new_func.at[0, 1].set(0.0)
                    new_func = new_func.at[1, 1].set(1.5)   # beneficial
                
                # Fill remaining with small noise (shared for all cases)
                for k in range(D):
                    for l in range(D):
                        if (k, l) not in [(0,0), (1,0), (0,1), (1,1)]:
                            noise_scale = 0.05 if not use_reciprocal else 0.02
                            new_func = new_func.at[k, l].set(noise_scale * np.random.randn())
                
                if hasattr(subgraph.edge_functions[edge], 'value'):
                    subgraph.edge_functions[edge].value = new_func
                else:
                    subgraph.edge_functions[edge] = new_func
        
        fg.apply_weight_fn(weight_fn_name)
        # Regenerate the evaluate functions for all subgraphs
        for subgraph in functional_subgraphs:
            if hasattr(subgraph, 'prepare_evaluate_jax'):
                subgraph.prepare_evaluate_jax()