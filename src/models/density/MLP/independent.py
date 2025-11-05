from typing import List, Tuple, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

from src.models.density.base import DensityModel
from src.models.density.MLP.base import UnconditionalDensityVector

class IndependentDensityMLP(DensityModel, nnx.Module):
    """
    MLP-based density model for a graph with no edges.
    i.e., a collection of independent unconditional models.
    """
    
    def __init__(self, n_states_list: List[int], 
                 hidden_dims: List[int], 
                 rngs: jax.random.PRNGKey):
        """
        Args:
            n_states_list: Number of possible states for each variable
            hidden_dims: List of hidden layer dimensions for the MLPs
            rngs: Random number generators for initialization
        """
        self.n_nodes = len(n_states_list)
        self.n_states_list = n_states_list

        # Create individual models for each node (no vmap)
        self.models = [None for _ in range(self.n_nodes)]
        model_rngs = jax.random.split(rngs, self.n_nodes)

        for i in range(self.n_nodes):
            self.models[i] = UnconditionalDensityVector(n_states_list[i], rngs=nnx.Rngs(model_rngs[i]))


    def _sample_node(self, node: int, rng: jax.random.PRNGKey = None) -> Tuple[int, jax.Array]:
        """Sample a value for a single node given previous samples."""
        log_probs = self.models[node]()
        
        if rng is not None:
            sample = jax.random.categorical(rng, log_probs)
        else:
            sample = jnp.argmax(log_probs)
            
        return sample.astype(jnp.int32), log_probs[sample]

    def _sample(self, rng: jax.random.PRNGKey = None) -> Tuple[jnp.ndarray, float]:
        """Sample a complete assignment from the model."""
        samples = jnp.zeros(self.n_nodes, dtype=jnp.int32)
        log_prob = 0.0
        node_rng = None
        
        # Sample nodes in order (since they're independent)
        for node in range(self.n_nodes):
            if rng is not None: rng, node_rng = jax.random.split(rng)
            node_sample, node_log_prob = self._sample_node(node, rng=node_rng)
            samples = samples.at[node].set(node_sample)
            log_prob += node_log_prob
            
        return samples, log_prob
    
    @partial(jax.vmap, in_axes=(None, 0)) # NOTE: should be ok b/c model is static.
    def sample(self, rng: jax.random.PRNGKey = None) -> Tuple[jnp.ndarray, float]:
        """Sample a complete assignment from the model."""
        return self._sample(rng)
    
    def sample_mode(self) -> Tuple[jnp.ndarray, float]:
        """Sample the mode of the model."""
        return self._sample(rng=None)
    
    def _node_likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a single complete assignment (non-vmapped)."""
        log_prob = jnp.zeros(self.n_nodes)
        for node in range(self.n_nodes):
            log_prob = log_prob.at[node].set(self.models[node]()[x[node]])
        
        return log_prob / x.shape[-1] # per-position likelihood
    
    @partial(nnx.vmap, in_axes=(None, 0))
    def node_likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a single complete assignment (vmapped over samples)."""
        return self._node_likelihood(x)

    @partial(nnx.vmap, in_axes=(None, 0)) # NOTE: model not static, need nnx.vmap (?)
    def likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment (vmapped over samples)."""
        return self._node_likelihood(x).sum()
    
    @partial(nnx.vmap, in_axes=(None, 0, None)) # NOTE: model not static, need nnx.vmap (?)
    def node_weighted_likelihood(self, x: jnp.ndarray, Q: jnp.ndarray) -> float:
        """Compute the weighted log-likelihood of a complete assignment."""
        # Q[node, x_node] for independent models (no parent)
        weights = Q[jnp.arange(self.n_nodes), x]
        node_logps = self._node_likelihood(x)
        return jnp.dot(weights, node_logps) / x.shape[-1] # mean likelihood