import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Optional, List
from functools import partial

from src.models.blocks import TransformerBlock
from src.models.density.base import DensityModel
from src.decomposition.graphs import Tree

class TreeDensityTransformer(DensityModel, nnx.Module):
    def __init__(self, tree: Tree, n_states: List[int], d_model: int = 64, n_heads: int = 4, d_ff: int = 128, n_layers: int = 2, *, rngs):
        self.n_states = n_states
        self.tree = tree
        self.root = tree.root
        self.node_order = tuple(tree.node_order)
        self.parents = tuple(tree.parents)
        self.children = tree.children
        self.n_nodes = len(self.node_order)
        
        self.SOS_TOKEN_ID = max(n_states)  # start-of-sequence token for root
        self.mod_states = [1] + [D+1 for D in n_states] # +1 for SOS token, and extra position.
        self.alphabet_size = max(self.mod_states) # +1 for SOS token accounted for
        self.n_layers = n_layers
        # TODO: should these actually be learned? I think DADO requires that params are SEPARATE between nodes.
        rngs, *model_rngs = jax.random.split(rngs, 1 + 3)
        self.token_embed = nnx.Embed(
            num_embeddings=self.alphabet_size, features=d_model, rngs=nnx.Rngs(model_rngs[0])
        )
        self.pos_embed = nnx.Param( # SOS token + nodes in topological order
            jax.random.normal(model_rngs[1], (len(self.mod_states)-1, d_model))
        )
        self.output_proj = nnx.Linear(d_model, self.alphabet_size, rngs=nnx.Rngs(model_rngs[-1]))

        layer_rngs = jax.random.split(rngs, n_layers)
        @nnx.vmap( # (in_axes defaults to None → broadcast)
            out_axes=nnx.StateAxes({
                nnx.Param:    0,
                nnx.RngState: 0,
                ...: None,
            })
        )
        def create_block(key):
            return TransformerBlock(d_model, n_heads, d_ff, rngs=nnx.Rngs(key))
        self.blocks = create_block(layer_rngs)

        self.parent_queries = self.get_parent_queries()

    @staticmethod
    @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
    def scan_blocks_train(block, carry):
        x, mask = carry
        return nnx.remat(block, static_argnums=(2, 3))(
            x, x, mask=mask, decode=False
        ), mask
    
    def __call__(self, x):
        """Process input tokens through the transformer model.

        Args:
            x: Query tokens of shape (B, L) containing categorical values (SOS/parents of each pos.)

        Returns:
            Logits of shape (B, L, alphabet_size) for **current** token prediction
        """
        x_emb = self.token_embed(x) + self.pos_embed[None, :, :]  # (B, L, D_model)
        mask = self._make_tree_mask(x, self.parents)
        x_emb = self.scan_blocks_train(self.blocks, (x_emb, mask))[0]
        return self.output_proj(x_emb)  # Return logits: (B, L, alphabet_size)
    
    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def _make_tree_mask(x, parents):
        """Create mask based on tree structure - nodes can only attend to SOS and their ancestors."""
        batch_size, seq_len = x.shape
        mask = jnp.zeros((batch_size, seq_len, seq_len), dtype=jnp.bool_)
        
        for i in range(seq_len):
            # Self-attention; this covers SOS token case for root clique first var.
            mask = mask.at[:, i, i].set(True)
            
            # Allow attention to parent
            parent_node = parents[i]  # parent in original order
            if parent_node != -1:  # If not root
                mask = mask.at[:, i, parent_node + 1].set(True)
        
        return mask[:, None, :, :] # Add singleton dimension for multi-head attention
    
    def get_parent_queries(self) -> jnp.ndarray:
        parent_inds = jnp.zeros((len(self.parents),)).astype(jnp.int32) # node index to its parent index
        for node in range(len(self.parents)):
            parent = self.parents[node]
            if parent == -1: # root has SOS as parent.
                parent_inds = parent_inds.at[node].set(0)
            else:
                parent_inds = parent_inds.at[node].set(parent + 1)
        return parent_inds

    @staticmethod
    @partial(jax.vmap, in_axes=(0, 1, -1)) # vmap over n_nodes
    def log_probs_node(n_states: int, logits: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        mask = jnp.arange(logits.shape[-1]) < n_states  # (max_states,)
        log_probs = jax.nn.log_softmax(
            jnp.where(mask, logits, -jnp.inf) # invalid states = -inf
        ) # (B, n_states)
        return log_probs[jnp.arange(x.shape[0]), x] # (B,); maintains original order of x.

    def node_likelihood(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute likelihood per node position (analogous to _likelihood in MLP version).
        
        Returns:
            node_log_probs: (B, n_nodes) log probabilities for each node
        """
        x_with_sos = jnp.concatenate([
            jnp.full(shape=[x.shape[0], 1], fill_value=self.SOS_TOKEN_ID, dtype=jnp.int32), 
            x
        ], axis=1) # shape (B, L+1)
        # SOS won't necessarily be first now.
        queries = x_with_sos[:, self.parent_queries] # rearrange s.t. each position has its parent token. shape (B, L)
        # NOTE: each position of logit is *next-token* prediction. Use SOS index, and not -1 index.
        logits = self(queries)  # shape (B, L, alphabet_size)
        # NOTE: softmax only over valid alphabet @ each position.
        return self.log_probs_node(jnp.array(self.n_states), logits, x).T / x.shape[-1] # shape (B, L)

    def node_weighted_likelihood(self, x: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
        """Compute likelihood per node position (analogous to _likelihood in MLP version).
        
        Returns:
            node_log_probs: (B, n_nodes) log probabilities for each node, weighted by Q
        """
        # Extract likelihoods for each node position
        node_log_probs = self.node_likelihood(x) # shape (B, L)

        # Extract weights (vectorized)
        parent_indices = jnp.array(self.parents).at[self.root].set(0)
        batch_indices = jnp.arange(x.shape[0])[:, None]  # (B, 1)
        parent_vals = x[:, parent_indices]  # (B, n_nodes)
        weights = Q[batch_indices, parent_vals, x]  # (B, n_nodes)
        
        return (node_log_probs * weights).sum(axis=-1) # mean likelihood

    def _node_weighted_likelihood(self, x: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
        """Compute likelihood per node position (analogous to _likelihood in MLP version).
        
        Returns:
            node_log_probs: (B, n_nodes) log probabilities for each node, weighted by Q
        """
        # Extract likelihoods for each node position
        node_log_probs = self.node_likelihood(x) # shape (B, L)

        # Extract weights (vectorized)
        parent_indices = jnp.array(self.parents).at[self.root].set(0)
        batch_indices = jnp.arange(x.shape[0])[:, None]  # (B, 1)
        parent_vals = x[:, parent_indices]  # (B, n_nodes)
        weights = Q[batch_indices, parent_vals, x]  # (B, n_nodes)
        
        return node_log_probs, weights # mean likelihood
    
    
    def sample_node(self, x, node_index, init_cache=False):
        """Sample a single node from the transformer model. In topological order.

        Args:
            x: Previous tokens of shape (B, L_current) containing categorical values  
            node_index: Index of the node being predicted (0-based, 0=root after SOS)

        Returns:
            Logits of shape (B, L, n_states) for the next token prediction
        """
        assert node_index < len(self.n_states), "node_index must be less than the number of nodes"
        x_emb = self.pos_embed[None, node_index, :] + self.token_embed(x)  # (B, 1, D_model)
        # TODO: may still be wrong; unsure how cache works. will it work if decoding order isn't 0-->L?
        if init_cache:
            nnx.vmap(
                lambda b: b.attention.init_cache(x_emb.shape, dtype=jnp.bfloat16)
            )(self.blocks)
        
        for i in range(self.n_layers): # can't scan b/c of init cache (different trace levels)
            x_emb = nnx.vmap(
                lambda b: b(
                    x_emb.astype(jnp.bfloat16),
                    x_emb.astype(jnp.bfloat16),
                    mask=None,
                    decode=True
                )
            )(self.blocks)[i] # discard all other layers :(
            
        return self.output_proj(x_emb)[
            ..., :self.n_states[node_index]
        ].squeeze().astype(jnp.float32)  # Return logits: (B, ~~1~~, n_states@node)
    
    def sample(self, rng: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Sample a batch of sequences from the model in topological order. 
        Inputs:
            rng: random number generator -- shape is n_samples. or None.
        Returns:
            samples: (n_samples, n_nodes) 
            log_prob: (n_samples,)
        '''
        n_samples = rng.shape[0] if rng is not None else 1
        samples = jnp.zeros((n_samples, self.n_nodes,), dtype=jnp.int32)
        log_probs = jnp.zeros((n_samples,))
        # SOS won't necessarily be first now.
        context = jnp.zeros((n_samples, self.n_nodes,), dtype=jnp.int32)
        context = context.at[:, self.root].set(self.SOS_TOKEN_ID) # SOS token for root.

        # Sample nodes in topological order
        for i, pos in enumerate(self.node_order): # TODO: can we jax.lax.scan?
            if i == 0: assert pos == self.root, "Root node should be first in topological order"
            # NOTE: disallow SOS token (last logit)
            logits = self.sample_node(context[:,pos][:,None], pos, init_cache=(i==0))[...,:-1]  
            probs = jax.nn.log_softmax(logits) # shape (n_samples, n_states[node])
            if rng is not None:
                rng, srng = jax.vmap(lambda k: jax.random.split(k, 2))(rng).transpose((1, 0, 2))
                tokens = jax.vmap(jax.random.categorical)(srng, probs).squeeze() # shape (n_samples,)
            else:
                tokens = jax.vmap(jnp.argmax)(probs).squeeze() # shape (n_samples,)
            samples = samples.at[:,pos].set(tokens)
            if len(self.children[pos]) > 0:
                context = context.at[:,jnp.array(self.children[pos])].set(tokens[:,None])
            log_probs = log_probs + probs[jnp.arange(n_samples), tokens]
            
        return samples, log_probs
    
    def sample_mode(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Sample the mode of the model.
        Returns:
            samples: (n_nodes,)
            log_prob: float
        '''
        samples, log_probs = self.sample()
        return samples.squeeze(), log_probs.squeeze()
    
    def likelihood(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute likelihood of samples given the tree structure."""
        return self.node_likelihood(x).sum(axis=1) # pre-normalized by length

    def _likelihood(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute likelihood of samples given the tree structure."""
        return self.node_likelihood(x)

    def fast_likelihood(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute likelihood of samples given the tree structure.
        Non-diff.
        """
        # Convert to pure function to avoid trace level conflicts
        graphdef, state = nnx.split(self)
        
        @jax.jit
        def pure_likelihood(state, x):
            # Recreate the model as a pure function
            model = nnx.merge(graphdef, state)
            
            x_with_sos = jnp.concatenate([
                jnp.full(shape=[x.shape[0], 1], fill_value=model.SOS_TOKEN_ID, dtype=jnp.int32), 
                x
            ], axis=1) # shape (B, L+1)
            # SOS won't necessarily be first now.
            queries = x_with_sos[:, model.parent_queries] # rearrange s.t. each position has its parent token. shape (B, L)
            # NOTE: each position of logit is *next-token* prediction. Use SOS index, and not -1 index.
            
            x_emb = model.token_embed(queries) + model.pos_embed[None, :, :]  # (B, L, D_model)
            mask = model._make_tree_mask(queries, model.parents)
            
            # Use the original scan approach since we're now in a pure function
            x_emb = model.scan_blocks_train(model.blocks, (x_emb, mask))[0]
            logits = model.output_proj(x_emb)  # Return logits: (B, L, alphabet_size)
            
            # NOTE: softmax only over valid alphabet @ each position.
            return model.log_probs_node(jnp.array(model.n_states), logits, x).T / x.shape[-1] # shape (B, L)
        
        return pure_likelihood(state, x)