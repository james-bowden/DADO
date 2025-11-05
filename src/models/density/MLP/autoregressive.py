from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

from src.models.density.base import DensityModel
from src.models.density.MLP.base import UnconditionalDensityVector, ConditionalDensityMLP


class AutoregressiveDensityMLP(DensityModel, nnx.Module):
    """MLP that models a joint distribution using an autoregressive factorization."""
    
    def __init__(
            self, 
            n_states_list: List[int], 
            hidden_dims: List[int], 
            rngs: jax.random.PRNGKey,
        ):
        """
        Args:
            n_states_list: Number of possible states for each variable
            hidden_dims: List of hidden layer dimensions for the MLPs
            rngs: Random number generators for initialization
        """
        self.n_nodes = len(n_states_list)
        self.n_states_list = n_states_list
        
        self.models = [None for _ in range(self.n_nodes)]
        model_rngs = jax.random.split(rngs, self.n_nodes)
        
        self.models[0] = UnconditionalDensityVector(n_states_list[0], rngs=nnx.Rngs(model_rngs[0]))
        # Subsequent variables are conditioned on all previous variables
        for i in range(1, self.n_nodes):
            # Input dimension is sum of previous variables' states
            input_dim = sum(n_states_list[:i])
            self.models[i] = ConditionalDensityMLP(input_dim, n_states_list[i], hidden_dims, rngs=nnx.Rngs(model_rngs[i]))
    
    
    def _sample_node(self, node: int, 
                     prev_states_oh: jax.Array = None, 
                     rng: jax.random.PRNGKey = None) -> Tuple[int, jax.Array]:
        """Sample a value for a single node given previous states."""
        if prev_states_oh is None:  # First variable
            log_probs = self.models[0]()
        else:  # Subsequent variables
            log_probs = self.models[node](prev_states_oh)
            
        if rng is not None:
            sample = jax.random.categorical(rng, log_probs)
        else:
            sample = jnp.argmax(log_probs)

        return sample.astype(jnp.int32), log_probs[sample]
    
    # TODO: allow to sample n together -- allocate memory at once? or us scan? etc.
    def _sample(self, rng: jax.random.PRNGKey = None) -> Tuple[jnp.ndarray, float]:
        """Sample a complete assignment from the model."""
        samples = jnp.zeros((self.n_nodes,), dtype=jnp.int32)
        log_prob = 0.0
        crng = None
        
        # Sample first variable
        if rng is not None: rng, crng = jax.random.split(rng)
        first_sample, first_log_probs = self._sample_node(0, rng=crng)
        samples = samples.at[0].set(first_sample)
        log_prob += first_log_probs
        
        # Sample remaining variables in order, accumulating prev_states
        prev_states = jax.nn.one_hot(first_sample, self.n_states_list[0])
        for node in range(1, self.n_nodes):
            # Sample current variable
            if rng is not None: rng, crng = jax.random.split(rng)
            node_sample, node_log_probs = self._sample_node(node, prev_states, rng=crng)
            samples = samples.at[node].set(node_sample)
            log_prob += node_log_probs
            
            # Update prev_states by concatenating the new one-hot state
            if node < self.n_nodes - 1:  # Don't need to update for the last variable
                new_state = jax.nn.one_hot(node_sample, self.n_states_list[node])
                prev_states = jnp.concatenate([prev_states, new_state])
            
        return samples, log_prob
    
    @partial(jax.vmap, in_axes=(None, 0)) # NOTE: should be ok b/c model is static.
    def sample(self, rng: jax.random.PRNGKey = None) -> Tuple[jnp.ndarray, float]:
        """Sample a complete assignment from the model."""
        return self._sample(rng)
    
    def sample_mode(self) -> Tuple[jnp.ndarray, float]:
        """Sample the mode of the model."""
        return self._sample(rng=None)

    # Could be done in parallel but I don't know how to do efficiently.
    def _likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a single complete assignment."""
        log_probs = jnp.zeros(x.shape[0])
        log_probs = log_probs.at[0].set(self.models[0]()[x[0]])
        
        # Create all one-hot encodings at once since we have the full assignment
        one_hot_states = [jax.nn.one_hot(x[i], self.n_states_list[i]) for i in range(self.n_nodes)]
        prev_states = jnp.concatenate(one_hot_states) # (sum(n_states_list),)
        
        for node in range(1, self.n_nodes): # TODO: could be a scan. 
            # could also be parallel, but models are different sizes so not clean.
            up_to_node = sum(self.n_states_list[:node])
            log_probs = log_probs.at[node].set(self.models[node](prev_states[:up_to_node])[x[node]])

        return log_probs / x.shape[-1] # normalize by length

    @partial(nnx.vmap, in_axes=(None, 0)) # NOTE: model not static, need nnx.vmap (?)
    def likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment."""
        return self._likelihood(x).sum() # pre-normalized by length
    
    @partial(jax.vmap, in_axes=(None, 0)) # used for training (density estimation), ~~so model static, ok~~
    def fast_likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment."""
        return self._likelihood(x).sum() # pre-normalized by length
    

class ConditionalAutoregressiveDensityMLP(DensityModel, nnx.Module):
    """MLP that models a joint distribution using autoregressive factorization, conditioned on parent variables."""
    
    def __init__(self, 
                 n_states_list: List[int], 
                 n_parent_states_list: List[int], 
                 hidden_dims: List[int], 
                 rngs: jax.random.PRNGKey,
                 ):
        """
        Args:
            n_states_list: Number of possible states for each variable in this clique
            n_parent_states_list: Number of possible states for the parent clique (conditioning variables)
            hidden_dims: List of hidden layer dimensions for the MLPs
            rngs: Random number generators for initialization
        """
        self.n_nodes = len(n_states_list)
        self.n_states_list = n_states_list
        self.n_parent_states_list = n_parent_states_list
        self.n_parent_states = sum(n_parent_states_list)
        
        # Create models for each position in the autoregressive chain
        self.models = [None for _ in range(self.n_nodes)]
        model_rngs = jax.random.split(rngs, self.n_nodes)
        
        # First variable is conditioned only on parent clique
        self.models[0] = ConditionalDensityMLP(self.n_parent_states, n_states_list[0], hidden_dims, rngs=nnx.Rngs(model_rngs[0]))
        
        # Subsequent variables are conditioned on parent clique + all previous variables in this clique
        for i in range(1, self.n_nodes):
            # Input dimension is parent states + sum of previous variables' states
            input_dim = self.n_parent_states + sum(n_states_list[:i])
            self.models[i] = ConditionalDensityMLP(input_dim, n_states_list[i], hidden_dims, rngs=nnx.Rngs(model_rngs[i]))
    
    
    def _sample_node(self, node: int,
                     parent_states_oh: jax.Array, 
                     prev_states_oh: jax.Array = None, 
                     rng: jax.random.PRNGKey = None) -> Tuple[int, jax.Array]:
        """Sample a value for a single node given parent and previous states."""
        if node == 0:  # First variable - only conditioned on parent
            log_probs = self.models[0](parent_states_oh)
        else:  # Subsequent variables - conditioned on parent + previous vars
            conditioning = jnp.concatenate([parent_states_oh, prev_states_oh])
            log_probs = self.models[node](conditioning)
            
        if rng is not None:
            sample = jax.random.categorical(rng, log_probs)
        else:
            sample = jnp.argmax(log_probs)

        return sample.astype(jnp.int32), log_probs[sample]
    
    def _sample(self, parent_states: jax.Array, rng: jax.random.PRNGKey = None) -> Tuple[jnp.ndarray, float]:
        """Sample a complete assignment from the conditional autoregressive model."""
        samples = jnp.zeros((self.n_nodes,), dtype=jnp.int32)
        log_prob = 0.0
        crng = None

        parent_states_oh = jnp.concatenate(
            [jax.nn.one_hot(parent_states[i], n_p_states) for i, n_p_states in enumerate(self.n_parent_states_list)]
        ) # (sum(n_parent_states_list),)

        sample_node = lambda node, prev_states_oh, rng: self._sample_node(
            node, parent_states_oh, prev_states_oh, rng
        )
        
        # Sample first variable
        if rng is not None: rng, crng = jax.random.split(rng)
        first_sample, first_log_prob = sample_node(0, None, crng)
        samples = samples.at[0].set(first_sample)
        log_prob += first_log_prob
        
        # Sample remaining variables in order, accumulating prev_states
        prev_states_oh = jax.nn.one_hot(first_sample, self.n_states_list[0])
        for node in range(1, self.n_nodes):
            # Sample current variable
            if rng is not None: rng, crng = jax.random.split(rng)
            node_sample, node_log_prob = sample_node(node, prev_states_oh, crng)
            samples = samples.at[node].set(node_sample)
            log_prob += node_log_prob
            
            # Update prev_states_oh by concatenating the new one-hot state
            if node < self.n_nodes - 1:  # Don't need to update for the last variable
                new_state = jax.nn.one_hot(node_sample, self.n_states_list[node])
                prev_states_oh = jnp.concatenate([prev_states_oh, new_state])
            
        return samples, log_prob
    
    @partial(jax.vmap, in_axes=(None, 0, 0)) # NOTE: should be ok b/c model is static.
    def sample(self, parent_states: jax.Array, rng: jax.random.PRNGKey = None) -> Tuple[jnp.ndarray, float]:
        """Sample a complete assignment from the conditional autoregressive model."""
        return self._sample(parent_states, rng)
    
    def sample_mode(self, parent_states: jax.Array = None) -> Tuple[jnp.ndarray, float]:
        """Sample the mode of the conditional autoregressive model."""
        return self._sample(parent_states, rng=None)

    @partial(nnx.vmap, in_axes=(None, 0, 0)) # NOTE: model not static, need nnx.vmap (?)
    def likelihood(self, x: jnp.ndarray, parent_states: jax.Array) -> float:
        """Compute the log-likelihood of a complete assignment given parent state."""
        return self._likelihood(x, parent_states).sum() # pre-normalized by length

    @partial(jax.vmap, in_axes=(None, 0, 0)) # used for training (density estimation), ~~so model static, ok~~
    def fast_likelihood(self, x: jnp.ndarray, parent_states: jax.Array) -> float:
        """Compute the log-likelihood of a complete assignment given parent state."""
        return self._likelihood(x, parent_states).sum() # pre-normalized by length

    def _likelihood(self, x: jnp.ndarray, parent_states: jax.Array) -> float:
        """Compute the log-likelihood of a complete assignment given parent states."""
        parent_states_oh = jnp.concatenate(
            [jax.nn.one_hot(parent_states[i], n_p_states) for i, n_p_states in enumerate(self.n_parent_states_list)]
        ) # (sum(n_parent_states_list),)
        prev_states_oh = [jax.nn.one_hot(x[i], self.n_states_list[i]) for i in range(self.n_nodes)]
        conditioning = jnp.concatenate([parent_states_oh] + prev_states_oh) # (sum(n_parent_states_list) + sum(n_states_list),)

        log_probs = jnp.zeros(x.shape[0])
        log_probs = log_probs.at[0].set(self.models[0](parent_states_oh)[x[0]])
        
        for node in range(1, self.n_nodes):
            up_to_node = self.n_parent_states + sum(self.n_states_list[:node])
            log_probs = log_probs.at[node].set(self.models[node](conditioning[:up_to_node])[x[node]])
        
        return log_probs / x.shape[-1] # normalize by length
