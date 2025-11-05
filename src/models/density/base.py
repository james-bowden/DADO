from abc import ABC, abstractmethod
from typing import Tuple
import jax
import jax.numpy as jnp

class DensityModel(ABC):
    """Base class for density models over discrete spaces."""
    
    @abstractmethod
    def sample(self, rng: jax.random.PRNGKey = None) -> Tuple[jnp.ndarray, float]:
        """
        Sample a complete assignment from the model.
        
        Args:
            rng: Optional random number generator key
            
        Returns:
            Tuple of (sample, log_probability)
        """
        pass
    
    @abstractmethod
    def likelihood(self, x: jnp.ndarray) -> float:
        """
        Compute the log-likelihood of a complete assignment.
        
        Args:
            x: Array of shape (n_variables,) containing the assignment
            
        Returns:
            Log-likelihood of the assignment
        """
        pass
    
    def probability(self, x: jnp.ndarray) -> float:
        """
        Compute the probability of a complete assignment.
        
        Args:
            x: Array of shape (n_variables,) containing the assignment
            
        Returns:
            Probability of the assignment
        """
        return jnp.exp(self.likelihood(x)) 