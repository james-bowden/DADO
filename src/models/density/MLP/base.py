import jax
import jax.numpy as jnp
from flax import nnx
from typing import List

from src.models.blocks import MLP

class UnconditionalDensityVector(nnx.Module):
    """Unconditional distribution p(x_root) as learnable vector of logits."""

    def __init__(self, n_states: int, *, rngs: nnx.Rngs):
        self.logits = nnx.Param(nnx.initializers.normal(0.02)(rngs.param(), (n_states,)))

    def __call__(self) -> jax.Array:
        return jax.nn.log_softmax(self.logits)

class UnconditionalDensityMLP(nnx.Module):
    """MLP that models p(x_root) for a root node in a tree.
    If you want to model multiple variables, multiply their numbers of states together."""
    
    def __init__(self, n_states_in: int, n_states_out: int, hidden_dims: List[int], *, rngs: nnx.Rngs):
        """
        Args:
            n_states: Number of possible states for the variable
            hidden_dims: List of hidden layer dimensions
            rngs: Random number generators for initialization
        """
        # Input is a scalar 1, output is logits over n_states
        self.mlp = MLP([n_states_in] + hidden_dims + [n_states_out], rngs=rngs)
        self.n_states_in = n_states_in
        self.rngs = rngs
        
    # NOTE: *args --> can be treated as conditional model, for jax formatting.
    def __call__(self, *args) -> jax.Array:
        # x should be a scalar 1
        logits = self.mlp(jnp.ones((self.n_states_in,)))
        return jax.nn.log_softmax(logits)  # Return log probabilities
    
    # TODO make unconditional density take in x but just ignore and use ones_like. so subclass conditional density.

class ConditionalDensityMLP(nnx.Module):
    """MLP that models p(x_child | x_parent) for a child node in a tree."""
    
    def __init__(self, n_states_in: int, n_states_out: int, hidden_dims: List[int], *, rngs: nnx.Rngs):
        """
        Args:
            n_states: Number of possible states for the variable
            hidden_dims: List of hidden layer dimensions
            rngs: Random number generators for initialization
        """
        # Input is one-hot encoded parent state, output is logits over n_states
        self.mlp = MLP([n_states_in] + hidden_dims + [n_states_out], rngs=rngs)
        self.rngs = rngs
        
    def __call__(self, parent_state: jax.Array) -> jax.Array:
        # parent_state should be one-hot encoded
        logits = self.mlp(parent_state)
        return jax.nn.log_softmax(logits)  # Return log probabilities
