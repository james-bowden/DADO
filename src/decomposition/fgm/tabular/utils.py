from typing import List, Tuple, Dict
import numpy as np
import jax.numpy as jnp

def random_edge_functions(edges: List[Tuple[int, int]], n_states_list: List[int]) -> Dict[Tuple[int, int], jnp.ndarray]:
    """
    Randomly generate edge functions for the graph.
    """
    return {
        edge: jnp.array(
            np.random.randn(n_states_list[edge[0]], n_states_list[edge[1]])
        ) for edge in edges
    }

def random_node_functions(n_states_list: List[int]) -> jnp.ndarray:
    """
    Randomly generate node functions for the graph.
    """
    return [jnp.array(np.random.randn(n_states)) for n_states in n_states_list]

def zero_node_functions(n_states_list: List[int]) -> jnp.ndarray:
    """
    Generate node functions with all zeros for the graph.
    
    Returns:
        Array of zeros of shape (n_nodes, n_states)
    """
    return [jnp.zeros(n_states) for n_states in n_states_list]

def zero_edge_functions(edges: List[Tuple[int, int]], n_states_list: List[int]) -> Dict[Tuple[int, int], jnp.ndarray]:
    """
    Generate edge functions with all zeros for the graph.
    """
    return {
        jnp.array(edge): jnp.zeros(
            n_states_list[edge[0]], n_states_list[edge[1]]
        ) for edge in edges
    }
