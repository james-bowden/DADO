import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

from src.opt.base import Optimizer

class RandomSearch(Optimizer):
    """Simple random search optimizer."""
    
    def precompute(self) -> None:
        """No pre-computation needed for random search."""
        pass
    
    def get_solutions(self, n_points: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate random solutions uniformly over the search space.
        
        Returns:
            Tuple containing:
                - Array of shape (n_points, len(n_states)) containing the generated solutions
                - Array of shape (n_points,) containing the objective values for each solution
        """
        samples = []
        for n_states in self.n_states:
            var_samples = jax.random.randint(
                self.rng,
                (n_points, 1),
                0,
                n_states
            )
            samples.append(var_samples)
            self.rng, _ = jax.random.split(self.rng)
        samples = jnp.concatenate(samples, axis=1)
        
        objective_values = self.objective.query(samples)
        
        return samples, objective_values
        
