from typing import List, Tuple, Optional, Union
import networkx as nx

import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

from src.models.density.base import DensityModel
from src.models.density.MLP.base import UnconditionalDensityVector, ConditionalDensityMLP
from src.decomposition.graphs import Tree

class TreeDensityMLP(DensityModel, nnx.Module):
    """MLP-based density model for a directed tree structure."""
    def __init__(self, tree: Tree, n_states_list: List[int], hidden_dims: List[int], 
                 rngs: jax.random.PRNGKey):
        """
        Args:
            tree: Tree containing the tree structure and node order
            n_states_list: Number of possible states for each variable
            hidden_dims: List of hidden layer dimensions for the MLPs
            rngs: Random number generators for initialization
        """
        self.tree = tree
        assert nx.is_tree(self.tree.graph), "Tree must be a directed tree"
        self.root = tree.root
        self.node_order = tree.node_order
        self.parents = tree.parents
        self.children = tree.children
        self.n_states_list = n_states_list
        self.n_nodes = len(self.node_order)

        # Create individual models for each node (no vmap)
        self.models = [None for _ in range(self.n_nodes)]
        model_rngs = jax.random.split(rngs, self.n_nodes)
        
        # Root node gets unconditional model
        root_index = self.node_order.index(self.root)
        root_node = self.node_order[root_index]
        self.models[root_index] = UnconditionalDensityVector(n_states_list[root_node], rngs=nnx.Rngs(model_rngs[root_index]))
        
        # All other nodes get conditional models
        for i in range(self.n_nodes):
            if i != root_index:
                node = self.node_order[i]
                parent_node = self.parents[node]
                self.models[i] = ConditionalDensityMLP(
                    n_states_list[parent_node], n_states_list[node], hidden_dims, rngs=nnx.Rngs(model_rngs[i])
                )

    def _sample_node(self, node: int, prev_samples: jax.Array = None, rng: jax.random.PRNGKey = None) -> Tuple[int, jax.Array]:
        """Sample a value for a single node given previous samples."""
        node_index = self.node_order.index(node)
        
        if node == self.root:  # Root node
            log_probs = self.models[node_index]()
        else:  # Non-root node - conditioned on parent
            parent_node = self.parents[node]
            parent_state = jax.nn.one_hot(prev_samples[parent_node], self.n_states_list[parent_node])
            log_probs = self.models[node_index](parent_state)
        
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
        
        # Sample nodes in topological order
        for node in self.node_order:
            if rng is not None: rng, node_rng = jax.random.split(rng)
                
            node_sample, node_log_prob = self._sample_node(node, samples, rng=node_rng)
            samples = samples.at[node].set(node_sample)
            log_prob += node_log_prob
            
        return samples, log_prob

    @partial(jax.vmap, in_axes=(None, None, 0))
    def sample_node(self, node: int, rng: jax.random.PRNGKey) -> Tuple[int, float]:
        return self._sample_node(node, rng=rng)

    @partial(jax.vmap, in_axes=(None, 0)) # model static so ok
    def sample(self, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, float]:
        """Sample a complete assignment from the model."""
        return self._sample(rng)
    
    def sample_mode(self) -> Tuple[jnp.ndarray, float]:
        """Sample the mode of the model."""
        return self._sample(rng=None)
    
    # TODO: can we jit this / jax scan?
    def _likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a single complete assignment (non-vmapped)."""
        node_logps = jnp.zeros(self.n_nodes)
        
        # Compute likelihood for each node in tree order
        for node in self.node_order:
            node_index = self.node_order.index(node)
            
            if node == self.root:  # Root node
                node_logps = node_logps.at[node].set(self.models[node_index]()[x[node]])
            else:  # Non-root node
                parent_node = self.parents[node]
                parent_state = jax.nn.one_hot(x[parent_node], self.n_states_list[parent_node])
                node_logps = node_logps.at[node].set(self.models[node_index](parent_state)[x[node]])
        
        return node_logps / x.shape[-1] # normalize by length

    @partial(nnx.vmap, in_axes=(None, 0)) # used for training (density estimation), ~~so model static, ok~~
    def likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment (vmapped over samples)."""
        return self._likelihood(x).sum() # pre-normalized by length

    @partial(jax.vmap, in_axes=(None, 0)) # used for training (density estimation), ~~so model static, ok~~
    def fast_likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment (vmapped over samples)."""
        return self._likelihood(x) # pre-normalized by length

    @partial(nnx.vmap, in_axes=(None, 0, None)) # NOTE: model not static, need nnx.vmap (?)
    def node_weighted_likelihood(
        self,
        x: jnp.ndarray,
        Q: jnp.ndarray, # Q[node, x_parent, x_node]; padded out to max n_states for each node
    ) -> float:
        """Compute the weighted log-likelihood of a complete assignment."""
        # For root node, use 0 as parent index since Q[root, :, x_root] are all equal
        parent_indices = jnp.array(self.parents).at[self.root].set(0)
        weights = Q[jnp.arange(self.n_nodes), x[parent_indices], x]
        
        node_logps = self._likelihood(x)
        return jnp.dot(node_logps, weights), node_logps # mean likelihood

    @partial(nnx.vmap, in_axes=(None, 0, None)) # NOTE: model not static, need nnx.vmap (?)
    def _node_weighted_likelihood(
        self,
        x: jnp.ndarray,
        Q: jnp.ndarray, # Q[node, x_parent, x_node]; padded out to max n_states for each node
    ) -> float:
        """Compute the weighted log-likelihood of a complete assignment."""
        # For root node, use 0 as parent index since Q[root, :, x_root] are all equal
        parent_indices = jnp.array(self.parents).at[self.root].set(0)
        weights = Q[jnp.arange(self.n_nodes), x[parent_indices], x] # (n_nodes,)
        
        node_logps = self._likelihood(x) # normalize by length
        return node_logps, weights 
    
    @partial(nnx.vmap, in_axes=(None, 0)) # NOTE: to be used for e.g., prior, where no training happening.
    def node_likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment for each node."""
        return self._likelihood(x) # normalize by length