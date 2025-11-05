import jax
import jax.numpy as jnp
from flax import nnx
from typing import List, Tuple
from functools import partial

from src.models.blocks import MLP, ZeroedMLP


class IndependentRegressionMLP(nnx.Module):
    """For L independent positions, each with n_states states, do scalar regression.
    For use as a value function."""
    
    def __init__(self, n_states: int, n_nodes: int, hidden_dims: List[int], *, rngs: jax.random.PRNGKey):
        """
        Args:
            n_states: Number of possible states for each variable
            n_nodes: Number of positions
            hidden_dims: List of hidden layer dimensions for the MLPs
            rngs: Random number generators for initialization
        """
        self.n_states = n_states
        self.n_nodes = n_nodes

        self.models = [
            MLP([n_states] + hidden_dims + [1], rngs=nnx.Rngs(r))
            for r in jax.random.split(rngs, n_nodes)
        ]

    def node_forward_single(self, node: int, state: int) -> float:
        """Return scalar for given state."""
        ohe = jax.nn.one_hot(state, self.n_states)
        return self.models[node](ohe).squeeze()

    @partial(nnx.vmap, in_axes=(None, None, 0))
    def node_forward(self, node: int, x: jnp.ndarray) -> float:
        """Return scalar for given state."""
        return self.node_forward_single(node, x[node])
    
    @partial(nnx.vmap, in_axes=(None, None, 0))
    def _node_forward(self, node: int, state: int) -> float:
        """Return scalar for given state."""
        self.node_forward_single(node, state)
    
    def forward(self):
        result = jnp.empty((self.n_nodes, self.n_states))
        states = jnp.arange(self.n_states)
        for node in range(self.n_nodes):
            result = result.at[node, :].set(self._node_forward(node, states))
        return result


class JointRegressionMLP(nnx.Module):
    """For L positions, each with n_states states, do scalar regression."""
    
    def __init__(self, 
                 n_states: List[int], 
                 hidden_dims: List[int], 
                 *, 
                 rngs: jax.random.PRNGKey,
                 scaled: bool = False, 
                 ): 
        """
        Args:
            n_states: Number of possible states for each variable
            hidden_dims: List of hidden layer dimensions for the MLPs
            rngs: Random number generators for initialization
        """
        self.n_states = n_states
        # NOTE: input is concatenated OHE of all states
        self.models = MLP(
            [sum(n_states)] + hidden_dims + [1], 
            rngs=nnx.Rngs(rngs), 
            scaled=scaled,
        )

    @staticmethod
    @partial(jax.vmap, in_axes=(None, 0))
    def _create_input_single(n_states: List[int], x: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([jax.nn.one_hot(x[i], s) for i, s in enumerate(n_states)])
    
    def forward(self, x: jnp.ndarray) -> float:
        """Return scalar for given sequence."""
        x_oh = JointRegressionMLP._create_input_single(self.n_states, x)
        return self.models(x_oh).squeeze()


class GraphRegressionMLP(nnx.Module):
    """For L positions, each with n_states states, do scalar regression."""
    
    def __init__(self, 
                 n_states: List[int], 
                 hidden_dims: List[int], 
                 *, 
                 rngs: jax.random.PRNGKey,
                 scaled: bool = False,
                 nonneg: bool = False,
                 ):
        """
        Even for cliques, n_states should be over the original variables. 
        Cliques (as well as edges) should be represented by multiple 1s in the bitvector.
        Args:
            n_states: Number of possible states for each variable
            hidden_dims: List of hidden layer dimensions for the MLPs
            rngs: Random number generators for initialization
        """
        self.n_states = tuple(n_states)
        # NOTE: input is [bitvector active node, OHE states for each node]
        #       A clique or an edge can be represented by multiple 1s in the bitvector.
        #       Any inactive node should have all 0s in the OHE states.
        #       Should start with 0 outputs for all inputs b/c final layer is 0-weighted.
        # NOTE: Apparently, needs to be ZeroedMLP, not MLP. Using scaled/nonneg also makes worse.
        self.models = ZeroedMLP(
            [len(n_states) + sum(n_states)] + hidden_dims + [1], 
            rngs=nnx.Rngs(rngs),
            scaled=scaled,
            nonneg=nonneg,
        )
    
    @staticmethod
    def _create_input_single(n_states: Tuple[int, ...], inds: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros((sum(n_states),))
        for i, D in enumerate(n_states):
            out_partial = jax.lax.select(
                inds[i],
                jax.nn.one_hot(x[i], D),
                jnp.zeros((D,))
            )
            start = sum(n_states[:i])
            out = out.at[start:start+D].set(out_partial)
        return out.astype(float)

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _forward(n_states: Tuple[int, ...], inds: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros((len(n_states) + sum(n_states),))
        out = out.at[:len(n_states)].set(inds.astype(float))
        out = out.at[len(n_states):].set(
            GraphRegressionMLP._create_input_single(n_states, inds, x)
        )
        return out
    
    def __call__(self, inds: jnp.ndarray, x: jnp.ndarray) -> float:
        """Return scalar for given sequence."""
        out = jax.vmap(
            lambda inds_single: self._forward(self.n_states, inds_single, x)
        )(inds)
        return self.models(out).squeeze()


class GraphRegressionMLPEmbedded(nnx.Module):
    """For L positions, each with n_states states, do scalar regression."""
    
    def __init__(self, 
                 n_states: List[int], 
                 hidden_dims: List[int], 
                 embedding_dim: int,
                 *, 
                 rngs: jax.random.PRNGKey,
                 scaled: bool = False,
                 nonneg: bool = False,
                 ):
        """
        Even for cliques, n_states should be over the original variables. 
        Cliques (as well as edges) should be represented by multiple 1s in the bitvector.
        Args:
            n_states: Number of possible states for each variable
            hidden_dims: List of hidden layer dimensions for the MLPs
            rngs: Random number generators for initialization
        """
        self.n_states = tuple(n_states)
        # NOTE: input is [bitvector active nodes, embedding of states for each node]
        #       A clique or an edge can be represented by multiple 1s in the bitvector.
        #       Any inactive node should have all 0s in the OHE states.
        #       Should start with 0 outputs for all inputs b/c final layer is 0-weighted.
        # Again, assuming that the states are the same across positions.
        self.embedding_dim = embedding_dim
        self.embedding = nnx.Embed(
            num_embeddings=max(n_states), 
            features=embedding_dim, 
            rngs=nnx.Rngs(rngs),
        )
        self.input_dim = len(n_states) * (1 + embedding_dim)
        # NOTE: Apparently, needs to be ZeroedMLP, not MLP. Using scaled/nonneg also makes worse.
        self.models = ZeroedMLP(
            [self.input_dim] + hidden_dims + [1], 
            rngs=nnx.Rngs(rngs),
            scaled=scaled,
            nonneg=nonneg,
        )
    
    def _create_input_single(self, inds: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        # Get embeddings for all x values at once (indexing operation)
        embeddings = self.embedding(x)  # (L, embedding_dim)
        masked_embeddings = jnp.where(
            inds[:, None],  # Broadcast inds to (L, 1)
            embeddings,
            jnp.zeros_like(embeddings)
        )
        return masked_embeddings.ravel().astype(float)  # (L * embedding_dim,)

    def _forward(self, inds: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros((self.input_dim,))
        out = out.at[:len(self.n_states)].set(inds.astype(float))
        out = out.at[len(self.n_states):].set(
            self._create_input_single(inds, x)
        )
        return out
    
    def __call__(self, inds: jnp.ndarray, x: jnp.ndarray) -> float:
        """Return scalar for given sequence."""
        out = jax.vmap(
            lambda inds_single: self._forward(inds_single, x)
        )(inds)
        return self.models(out).squeeze()