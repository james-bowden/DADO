from typing import List, Tuple, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

from src.models.density.base import DensityModel
from src.models.density.MLP.autoregressive import AutoregressiveDensityMLP, ConditionalAutoregressiveDensityMLP
from src.decomposition.graphs import JunctionTree

class JunctionTreeDensityCliqueMLPs(DensityModel, nnx.Module):
    """MLP-based density model for a directed junction tree structure."""
    
    def __init__(self, junction_tree: JunctionTree, 
                 n_states_list: List[int], hidden_dims: List[int], 
                 rngs: jax.random.PRNGKey):
        """
        Args:
            junction_tree: JunctionTree containing the junction tree structure and node order
            n_states_list: Number of possible states for each variable in the original graph
            hidden_dims: List of hidden layer dimensions for the MLPs
            rngs: Random number generators for initialization
        """
        self.tree = junction_tree
        self.root = self.tree.root
        self.node_order = self.tree.node_order
        self.parents = self.tree.parents
        self.children = self.tree.children
        self.n_states_list = n_states_list
        self.n_nodes = len(self.node_order)
        
        # Store junction tree structure
        self.index_to_nodes = [None for _ in range(self.n_nodes)]  # Maps clique index to set of original variable nodes
        
        # Create models - different sizes for different cliques
        self.models = [None for _ in range(self.n_nodes)]
        model_rngs = jax.random.split(rngs, self.n_nodes)
        
        for i in range(self.n_nodes):
            clique_vars = sorted(list(self.tree.index_to_nodes[i]))
            clique_n_states_list = [n_states_list[var] for var in clique_vars]
            
            if i == self.root:
                # Root clique uses autoregressive model (no conditioning)
                self.models[i] = AutoregressiveDensityMLP(
                    clique_n_states_list, hidden_dims, model_rngs[i],
                )
            else:
                # Non-root cliques use conditional autoregressive model
                parent_idx = self.parents[i]
                parent_clique_vars = sorted(list(self.tree.index_to_nodes[parent_idx]))
                parent_n_states_list = [n_states_list[var] for var in parent_clique_vars]
                
                # NOTE: try to account for overlap in vars, parent vars.
                clique_vars = [var for var in clique_vars if var not in parent_clique_vars]
                clique_n_states_list = [n_states_list[var] for var in clique_vars]
                
                self.models[i] = ConditionalAutoregressiveDensityMLP(
                    clique_n_states_list, parent_n_states_list, hidden_dims, model_rngs[i],
                )
            self.index_to_nodes[i] = clique_vars

    @staticmethod
    def _sample_clique(model, parent_states: Optional[jax.Array], rng: jax.random.PRNGKey = None) -> Tuple[jax.Array, float]:
        """Sample from a single clique model."""
        # NOTE: use single sample versions from model; their sample functions are vmap'd over rngs.
        if parent_states is None: # Root clique - use autoregressive model
            return model._sample(rng)
        else: # Non-root clique - use conditional autoregressive model
            return model._sample(parent_states, rng)

    @staticmethod
    def _sample(models,
                node_order: List[int],
                root: int,
                parents: List[int],
                index_to_nodes: List[set],
                rng: jax.random.PRNGKey) -> Tuple[jax.Array, float]:
        """Sample from the entire junction tree."""
        # Initialize with zeros - will be filled as we sample cliques
        n_original_vars = max(max(clique) for clique in index_to_nodes) + 1
        samples = jnp.zeros(n_original_vars, dtype=jnp.int32)
        total_logp = 0.0
        
        for clique_idx in node_order: # TODO: can we jax.lax.scan?
            clique_vars = index_to_nodes[clique_idx] # should already have no overlap
            # Determine parent state for conditioning
            if clique_idx == root:
                parent_states = None
            else:
                parent_idx = parents[clique_idx]
                parent_clique_vars = index_to_nodes[parent_idx]
                # Extract parent clique assignment from current samples
                parent_states = jnp.array([samples[var] for var in parent_clique_vars])
            
            # Sample from this clique
            if rng is not None:
                rng, clique_rng = jax.random.split(rng)
            else:
                clique_rng = None  # For mode sampling (argmax)
            clique_assignment, clique_logp = JunctionTreeDensityCliqueMLPs._sample_clique(
                models[clique_idx], parent_states, rng=clique_rng
            )
            
            # Update the overall assignment with this clique's variables
            for i, var in enumerate(clique_vars):
                samples = samples.at[var].set(clique_assignment[i])
            
            total_logp += clique_logp
        
        return samples, total_logp

    @partial(jax.vmap, in_axes=(None, 0)) # model static so ok
    def sample(self, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, float]:
        """Sample a complete assignment from the junction tree model."""
        # self.eval()
        result = JunctionTreeDensityCliqueMLPs._sample(
            self.models,
            self.node_order,
            self.root,
            self.parents,
            self.index_to_nodes,
            rng,
        )
        # self.train()
        return result
    
    def sample_mode(self) -> Tuple[jnp.ndarray, float]:
        """Sample the mode of the junction tree model."""
        self.eval()
        result = JunctionTreeDensityCliqueMLPs._sample(
            self.models,
            self.node_order,
            self.root,
            self.parents,
            self.index_to_nodes,
            rng=None,
        )
        self.train()
        return result
    
    def sample_original(self, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, float]:
        """Sample and return in original variable order (same as sample for this implementation)."""
        return self.sample(rng)

    def _likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment."""
        total_logp = jnp.zeros((x.shape[0], x.shape[-1])) # shape (n_samples, L)
        
        for clique_idx in self.node_order:
            clique_vars = self.index_to_nodes[clique_idx] # should already have no overlap
            
            if clique_idx == self.root:
                # Root clique
                logp = self.models[clique_idx]._likelihood(x[jnp.array(clique_vars)])
            else:
                # Non-root clique - need parent states (individual states, not mixed-radix)
                parent_idx = self.parents[clique_idx]
                parent_clique_vars = self.index_to_nodes[parent_idx]
                parent_states = x[jnp.array(parent_clique_vars)]
                
                logp = self.models[clique_idx]._likelihood(x[jnp.array(clique_vars)], parent_states)

            total_logp = total_logp.at[:, jnp.array(clique_vars)].set(logp)
        
        return total_logp

    @partial(nnx.vmap, in_axes=(None, 0)) # used for training (density estimation), ~~so model static, ok~~
    def likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment."""
        return self._likelihood(x).mean() # mean likelihood

    @partial(jax.vmap, in_axes=(None, 0)) # used for training (density estimation), ~~so model static, ok~~
    def fast_likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment."""
        logps = jnp.zeros(len(self.node_order))
        
        for clique_idx in self.node_order:
            clique_vars = self.index_to_nodes[clique_idx] # should already have no overlap
            
            # Compute likelihood for this clique
            if clique_idx == self.root:
                logp = self.models[clique_idx]._likelihood(x[jnp.array(clique_vars)])
            else:
                parent_idx = self.parents[clique_idx]
                parent_clique_vars = self.index_to_nodes[parent_idx]
                parent_states = x[jnp.array(parent_clique_vars)]
                
                logp = self.models[clique_idx]._likelihood(x[jnp.array(clique_vars)], parent_states)
            
            logps = logps.at[clique_idx].set(logp.sum(axis=-1))
        
        return logps

    @partial(nnx.vmap, in_axes=(None, 0)) # for use in e.g., prior, where no training happening.
    def clique_likelihood(
        self,
        x: jnp.ndarray,
    ) -> float:
        """Compute the weighted log-likelihood for junction tree using sample-based Q values."""
        logps = jnp.zeros(len(self.node_order))
        
        for clique_idx in self.node_order:
            clique_vars = self.index_to_nodes[clique_idx] # should already have no overlap
            
            # Compute likelihood for this clique
            if clique_idx == self.root:
                logp = self.models[clique_idx]._likelihood(x[jnp.array(clique_vars)])
            else:
                parent_idx = self.parents[clique_idx]
                parent_clique_vars = self.index_to_nodes[parent_idx]
                parent_states = x[jnp.array(parent_clique_vars)]
                
                logp = self.models[clique_idx]._likelihood(x[jnp.array(clique_vars)], parent_states)
            
            logps = logps.at[clique_idx].set(logp.sum(axis=-1))
        
        return logps

    @partial(nnx.vmap, in_axes=(None, 0, 0)) # NOTE: model not static, need nnx.vmap (?)
    def clique_weighted_likelihood(
        self,
        x: jnp.ndarray,
        Q: jnp.ndarray, # Q[n_cliques] - Q values for this sample
    ) -> float:
        """Compute the weighted log-likelihood for junction tree using sample-based Q values."""
        total_logp = 0.0
        
        for clique_idx in self.node_order: # NOTE: can't scan b/c models not vmap'd.
            clique_vars = self.index_to_nodes[clique_idx] # should already have no overlap
            
            # Get weight for this clique from sample-based Q array
            weight = Q[clique_idx] # NOTE: needs to be positive!!
            
            # Compute likelihood for this clique
            if clique_idx == self.root:
                logp = self.models[clique_idx]._likelihood(x[jnp.array(clique_vars)])
            else:
                parent_idx = self.parents[clique_idx]
                parent_clique_vars = self.index_to_nodes[parent_idx]
                parent_states = x[jnp.array(parent_clique_vars)]
                
                logp = self.models[clique_idx]._likelihood(x[jnp.array(clique_vars)], parent_states)
            
            total_logp += (weight * logp.sum(axis=-1))
        
        return total_logp # pre-normalized by length
    
class JunctionTreeDensityMLP(DensityModel, nnx.Module):
    def __init__(self, junction_tree: JunctionTree, n_states: List[int], hidden_dims: List[int], *, rngs):
        pass

def prod(l: List[int]) -> int:
    result = 1
    for x in l:
        result *= x
    return result