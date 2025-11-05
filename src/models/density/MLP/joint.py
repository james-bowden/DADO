from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

from src.models.density.base import DensityModel
from src.models.density.MLP.base import UnconditionalDensityVector

class JointDensityMLP(DensityModel, nnx.Module):
    """MLP that models a joint distribution over all variables independently."""
    
    def __init__(self, n_states_list: List[int], hidden_dims: List[int], *, rngs: nnx.Rngs):
        """
        Args:
            n_states_list: Number of possible states for each variable
            hidden_dims: List of hidden layer dimensions for the MLPs
            rngs: Random number generators for initialization
        """
        # raise NotImplementedError("JointDensityMLP is deprecated; use AutoregressiveDensityMLP instead.")
        self.n_nodes = len(n_states_list)
        self.n_states_list = n_states_list
        self.n_states = n_states_list[0] # assumes all states are the same, wrong.
        
        # Initialize an unconditional model over all variables
        # NOTE: this will cause issues with memory when n_nodes gets large.
        # should use AutoregressiveDensityMLP instead.
        all_vars = jnp.prod(jnp.array(n_states_list)).item()
        self.model = UnconditionalDensityVector(all_vars, rngs=nnx.Rngs(rngs))
        self.sample_shape = (self.n_states,) * self.n_nodes
    
    def _sample(self, rng: jax.random.PRNGKey = None) -> Tuple[int, jax.Array]:
        """Sample a value for all nodes."""
        log_probs = self.model()
            
        # Sample from the distribution -- this gives an index in [0, D^L)
        if rng is not None:
            sample = jax.random.categorical(rng, log_probs)
        else:
            sample = jnp.argmax(log_probs)
            
        # Convert the index to a sample
        sample = jnp.array(jnp.unravel_index(sample, self.sample_shape))
            
        return sample.astype(jnp.int32), log_probs[sample]
    
    @partial(jax.vmap, in_axes=(None, 0))
    def sample(self, rng: jax.random.PRNGKey = None) -> Tuple[jnp.ndarray, float]:
        """Sample a complete assignment from the model."""
        return self._sample(rng)

    def sample_mode(self) -> Tuple[jnp.ndarray, float]:
        """Sample the mode of the model."""
        return self._sample(rng=None)
    
    @partial(nnx.vmap, in_axes=(None, 0)) # NOTE: model not static, need nnx.vmap (?)
    def likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment."""
        # convert to a flat index; use clip mode
        flat_x = jnp.ravel_multi_index(x, self.sample_shape, mode='clip')
        return self.model()[flat_x]