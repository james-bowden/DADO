import jax
import jax.numpy as jnp
import numpy as np
from itertools import product
from typing import Tuple
from tqdm import trange

from src.opt.base import Optimizer


def cartesian_product(n_states: list[int]) -> jnp.ndarray:
    """Returns the Cartesian product of per-variable state ranges.

    Args:
        n_states: List of ints, each specifying the number of states for a variable.

    Returns:
        An array of shape (num_combinations, len(n_states)) where each row is a configuration.
    """
    grids = jnp.meshgrid(*[jnp.arange(s) for s in n_states], indexing="ij")
    return jnp.stack(grids, axis=-1).reshape(-1, len(n_states))

def cartesian_product_batch(n_states: list[int], start: int, batch_size: int) -> np.ndarray:
    """
    Efficiently compute a slice of the Cartesian product using mixed-radix
    base conversion.
    """
    n_states = np.array(n_states, dtype=np.int64)
    total = int(np.prod(n_states))
    batch_size = min(batch_size, total - start)

    idxs = np.arange(start, start + batch_size, dtype=np.int64)

    out = np.empty((batch_size, len(n_states)), dtype=np.int64)

    # Instead of re-dividing idxs for each variable,
    # peel off one "digit" at a time.
    for i in reversed(range(len(n_states))):
        out[:, i] = idxs % n_states[i]
        idxs //= n_states[i]

    return out

class ExhaustiveSearch(Optimizer):
    """Exhaustive search that evaluates all possible solutions."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_solutions = None
        self.sorted_indices = None
        self.values = None
        self.best_x = None
        self.best_val = None

    def precompute(self, use_fg: bool = True) -> None:
        """Generate and evaluate all possible solutions.
        
        Args:
            use_fg: Whether to use the functional graph representation of the objective.
            Only relevant if available.
        """
        # Generate all possible solutions
        if not use_fg or not hasattr(self.objective, 'fg'):
            N = np.prod(self.objective.D)
            if self.verbose: print(f"Warning: exhaustive search over {N} solutions. This may not be tractable.")
            
            # Evaluate all solutions
            batch_size = int(1e8)
            if N <= batch_size:
                self.all_solutions = cartesian_product(self.objective.D)
                self.values = self.objective.query(self.all_solutions)
            else:
                print(f"Batching {N} total points by {batch_size} ({(N/batch_size):.2f} batches).")
                values = []
                self.all_solutions = []
                for i in trange(0, N, batch_size):
                    batch = cartesian_product_batch(self.objective.D, i, batch_size)
                    values.append(np.array(self.objective.query(batch)))
                    self.all_solutions.append(batch)
                self.values = np.concatenate(values)
                self.all_solutions = np.concatenate(self.all_solutions)
        else: # do message passing on obj.fg!
            fg = self.objective.fg
            if len(fg.edges) == 0:
                # TODO: want to implement independent in terms of generalized graph (subgraphs w/ 1 node)
                raise NotImplementedError("Exhaustive search over functional graphs with no edges is not implemented.")
            else: # tree or JT
                V, Q, A = fg.message_passing()
                self.best_x = (-1) * np.ones((fg.n_nodes,)).astype(np.int64)
                self.best_val = 0.
                for node in fg.node_order:
                    parent = fg.parents[node]
                    if parent == -1:
                        self.best_x[node] = np.argmax(V[node])
                    else:
                        parent_state = self.best_x[parent]
                        # NOTE: doesn't matter if use Q or A. argmax is the same.
                        self.best_x[node] = np.argmax(A[node, parent_state, :])
                        self.best_val += fg.edge_functions[(parent, node)][parent_state, self.best_x[node]]
                    self.best_val += fg.node_functions[node][self.best_x[node]]
        
    def get_solutions(self, n_points: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return the best n_points solutions.
        
        Returns:
            Tuple containing:
                - Array of shape (n_points, n_variables) containing the generated solutions
                - Array of shape (n_points,) containing the objective values for each solution
        """
        if self.all_solutions is None or self.values is None:
            raise ValueError("Must call precompute() before get_solutions()")
        if self.sorted_indices is None:
            # Sort solutions by objective value
            self.sorted_indices = np.argsort(self.values, descending=True)
            
        # Return the top n_points solutions and their values
        top_indices = self.sorted_indices[:n_points]
        return jnp.array(self.all_solutions[top_indices]), jnp.array(self.values[top_indices])
    
    def get_best_solution(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return the best solution.
        
        Returns:
            Tuple containing:
                - Array of shape (n_variables) containing the generated solutions
                - Array of shape () containing the objective values for each solution
        """
        if self.best_x is not None and self.best_val is not None: # accomodate message passing
            return self.best_x, self.best_val
        if self.all_solutions is None or self.values is None:
            raise ValueError("Must call precompute() before get_best_solution()")
        if self.sorted_indices is None: # just compute max instead of sorting
            ind = np.argmax(self.values)
            self.best_x = jnp.array(self.all_solutions[ind])
            self.best_val = jnp.array(self.values[ind])
            return self.best_x, self.best_val
        else:
            self.best_x = jnp.array(self.all_solutions[self.sorted_indices[0]])
            self.best_val = jnp.array(self.values[self.sorted_indices[0]])
            return self.best_x, self.best_val
