import jax
import jax.numpy as jnp
from typing import Tuple

from src.opt.base import Optimizer
from src.opt.search.random import RandomSearch

class GreedySearch(Optimizer):
    """Greedy search that starts from random points and makes best single-variable changes."""
    
    def precompute(self) -> None:
        """No pre-computation needed for greedy search."""
        pass
    
    def _local_search(self, start_point: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        """
        Perform local search from a starting point until no better solution is found.
        
        Returns:
            Tuple containing:
                - The best solution found
                - The objective value of that solution
        """
        def cond_fun(state):
            _, _, _, _, improved = state
            return improved
            
        def body_fun(state):
            current, current_value, _, _, _ = state
            
            def scan_vars(carry, var):
                best_n, best_v = carry
                
                def body_fn(state_idx, carry):
                    best_n, best_v = carry
                    
                    # Create neighboring solution
                    neighbor = current.at[var].set(state_idx)
                    neighbor_value = self.objective.query(neighbor[None,:])[0]
                    
                    # Update best if this neighbor is better and different from current
                    is_better = jnp.logical_and(state_idx != current[var], neighbor_value > best_v)
                    new_best_n = jax.lax.select(is_better, neighbor, best_n)
                    new_best_v = jax.lax.select(is_better, neighbor_value, best_v)
                    
                    return (new_best_n, new_best_v)
                
                # Use fori_loop to iterate over states for this variable
                var_states = jnp.array(self.n_states)[var]
                final_carry = jax.lax.fori_loop(
                    0, 
                    var_states,
                    body_fn,
                    (best_n, best_v)
                )
                
                return final_carry, None
            
            (best_neighbor, best_value), _ = jax.lax.scan(
                scan_vars,
                (current, current_value),
                jnp.arange(len(self.n_states))
            )
            
            improved = best_value > current_value
            
            return best_neighbor, best_value, best_neighbor, best_value, improved
            
        # Initialize state with start_point
        start_value = self.objective.query(start_point[None,:])[0]
        init_state = (start_point, start_value, start_point, start_value, True)
        
        final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
        return final_state[0], final_state[1]
    
    def get_solutions(self, n_points: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform local search from n_points different random starting points.
        
        Returns:
            Tuple containing:
                - Array of shape (n_points, n_variables) containing the generated solutions
                - Array of shape (n_points,) containing the objective values for each solution
        """
        # Generate random starting points
        start_points = RandomSearch(
            n_states=self.n_states,
            objective_fn=self.objective,
            seed=self.seed
        ).get_solutions(n_points)[0]
        
        # Create a lambda that has self already bound
        search_fn = lambda x: self._local_search(x)
        
        # Vectorize the search function
        vectorized_search = jax.vmap(search_fn)
        
        # Run vectorized search across all start points -- may need to batch if gets too big.
        solutions, values = [], []
        for i in range(0, n_points, 1000):
            s, v = vectorized_search(start_points[i:i+1000])
            solutions.append(s)
            values.append(v)
        
        return jnp.concatenate(solutions, axis=0), jnp.concatenate(values, axis=0)
