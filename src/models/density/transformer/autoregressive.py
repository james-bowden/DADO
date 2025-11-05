import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Optional, List
from functools import partial

from src.models.blocks import TransformerBlock
from src.models.density.base import DensityModel

class AutoregressiveDensityTransformer(DensityModel, nnx.Module):
    def __init__(self, n_states: List[int], d_model: int = 64, n_heads: int = 4, d_ff: int = 128, n_layers: int = 2, *, rngs):
        self.n_states = n_states
        self.SOS_TOKEN_ID = max(n_states)  # start-of-sequence token. Allocate extra length, state below.
        self.mod_states = [1] + [D+1 for D in n_states] # +1 for SOS token, and extra position.
        self.alphabet_size = max(self.mod_states) # +1 for SOS token? No, accounted for in AutoregressiveDensityTransformer.
        self.n_layers = n_layers
        # TODO: should these actually be learned? I think DADO requires that params are SEPARATE between nodes.
        rngs, *model_rngs = jax.random.split(rngs, 1 + 3)
        self.token_embed = nnx.Embed(
            num_embeddings=self.alphabet_size, features=d_model, rngs=nnx.Rngs(model_rngs[0])
        )
        self.pos_embed = nnx.Param( # SOS token accounted for in AutoregressiveDensityTransformer.
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
            x: Input tokens of shape (B, L) containing categorical values, where B is batch size and L is sequence length

        Returns:
            Logits of shape (B, L, alphabet_size) for next token prediction
        """
        x_emb = self.token_embed(x) + self.pos_embed[None, :, :]  # (B, L, D_model)
        mask = nnx.make_causal_mask(x)
        x_emb = self.scan_blocks_train(self.blocks, (x_emb, mask))[0]  
        return self.output_proj(x_emb)  # Return logits: (B, L, alphabet_size), each next-token pred.
    
    def sample_node(self, x, node_index, init_cache=False):
        """Sample a single node from the transformer model.

        Args:
            x: Previous tokens of shape (B, L_current) containing categorical values
            node_index: Index of the node being predicted

        Returns:
            Logits of shape (B, L, n_states) for the next token prediction
        """
        assert node_index < len(self.n_states), "node_index must be less than or equal to the number of states"
        x_emb = self.pos_embed[None, node_index, :] + self.token_embed(x)  # (B, 1, D_model)
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
            ..., :self.n_states[node_index] # b/c predicting next token.
        ].squeeze().astype(jnp.float32)  # Return logits: (B, ~~1~~, n_states@node_index)
    
    def sample(self, rng: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Sample a batch of sequences from the model. 
        Inputs:
            rng: random number generator -- shape is n_samples. or None.
        Returns:
            samples: (n_samples, n_nodes)
            log_prob: (n_samples,)
        '''
        n_samples = rng.shape[0] if rng is not None else 1
        samples = jnp.zeros((n_samples, len(self.n_states)+1,), dtype=jnp.int32)
        samples = samples.at[:,0].set(self.SOS_TOKEN_ID) # context
        log_probs = jnp.zeros((n_samples,))

        for t in range(len(self.n_states)):
            logits = self.sample_node(samples[:,t][:,None], t, init_cache=(t==0)) # logits for position t+1
            probs = jax.nn.log_softmax(logits) # shape (n_samples, n_states@pos)
            if rng is not None:
                rng, srng = jax.vmap(lambda k: jax.random.split(k, 2))(rng).transpose((1, 0, 2))
                tokens = jax.vmap(jax.random.categorical)(srng, probs)
            else:
                tokens = jax.vmap(jnp.argmax)(probs)
            samples = samples.at[:,t+1].set(tokens) # result and next query
            log_probs = log_probs + probs[jnp.arange(n_samples), tokens]

        return samples[:,1:], log_probs # slice off SOS token
    
    def sample_mode(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Sample the mode of the model.
        Returns:
            samples: (n_nodes,)
            log_prob: float
        '''
        samples, log_probs = self.sample()
        return samples.squeeze(), log_probs.squeeze()

    @staticmethod
    @partial(jax.vmap, in_axes=(0, 1, -1)) # vmap over n_nodes
    def log_probs_node(n_states: int, logits: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        mask = jnp.arange(logits.shape[-1]) < n_states  # (max_states,)
        log_probs = jax.nn.log_softmax(
            jnp.where(mask, logits, -jnp.inf) # invalid states = -inf
        ) # (B, n_states)
        return log_probs[jnp.arange(x.shape[0]), x] # (B,)
    
    def _likelihood(self, x: jnp.ndarray) -> jnp.ndarray:
        queries = jnp.concatenate([
            jnp.full(shape=[x.shape[0], 1], fill_value=self.SOS_TOKEN_ID, dtype=jnp.int32), 
            x[:,:-1] # don't need to predict from last token.
        ], axis=1)
        logits = self(queries)  # shape (B, L, n_states+1)
        # NOTE: softmax only over valid alphabet @ each position.
        return self.log_probs_node(jnp.array(self.n_states), logits, x).T / x.shape[-1] # shape (B, L)
    
    def likelihood(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._likelihood(x).sum(axis=1) # shape (B,)

    def fast_likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment given parent state.
        Non-diff.
        """
        # Convert to pure function to avoid trace level conflicts
        graphdef, state = nnx.split(self)
        
        @jax.jit
        def pure_likelihood(state, x):
            queries = jnp.concatenate([
                jnp.full(shape=[x.shape[0], 1], fill_value=self.SOS_TOKEN_ID, dtype=jnp.int32), 
                x[:,:-1] # don't need to predict from last token.
            ], axis=1)
            
            # Recreate the model as a pure function
            model = nnx.merge(graphdef, state)
            x_emb = model.token_embed(queries) + model.pos_embed[None, :, :]  # (B, L, D_model)
            mask = nnx.make_causal_mask(queries)
            
            # Use the original scan approach since we're now in a pure function
            x_emb = model.scan_blocks_train(model.blocks, (x_emb, mask))[0]
            logits = model.output_proj(x_emb)  # Return logits: (B, L, alphabet_size)
            
            # NOTE: softmax only over valid alphabet @ each position.
            return (model.log_probs_node(jnp.array(model.n_states), logits, x).T / x.shape[-1]).sum(axis=1) # shape (B,)
        
        return pure_likelihood(state, x)
    

class ConditionalAutoregressiveDensityTransformer(DensityModel, nnx.Module):
    def __init__(self, n_states: List[int], n_parent_states_list: List[int], d_model: int = 64, n_heads: int = 4, d_ff: int = 128, n_layers: int = 2, *, rngs):
        self.n_states = n_states
        self.n_parent_states_list = n_parent_states_list
        assert sum(n_parent_states_list) > 0, "Use AutoregressiveDensityTransformer if you don't have any conditioning inputs"
        
        # NOTE: no SOS token since we always have conditioning inputs
        self.mod_states = n_states + n_parent_states_list
        self.alphabet_size = max(n_states + n_parent_states_list)
        self.n_layers = n_layers
        
        # TODO: should these actually be learned?
        rngs, *model_rngs = jax.random.split(rngs, 1 + 3)
        self.token_embed = nnx.Embed(
            num_embeddings=self.alphabet_size, features=d_model, rngs=nnx.Rngs(model_rngs[0])
        )
        self.pos_embed = nnx.Param( # parent states + child states (no SOS token)
            jax.random.normal(model_rngs[1], (len(n_parent_states_list) + len(n_states)-1, d_model))
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

    @staticmethod
    @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
    def scan_blocks_train(block, carry):
        q, k, mask = carry
        return nnx.remat(block, static_argnums=(2, 3))(
            q, k, mask=mask, decode=False
        ), k, mask
    
    def __call__(self, x):
        """Process input tokens through the transformer model.

        Args:
            x: Input tokens of shape (B, L) containing parent_states + child_states

        Returns:
            Logits of shape (B, L, alphabet_size) for next token prediction
        """
        parent_pos = len(self.n_parent_states_list)
        keys = self.token_embed(x) + self.pos_embed[None, :, :]  # (B, L, D_model)
        queries = keys[:, parent_pos-1:, :]
        mask = self._make_conditional_mask(x, parent_pos)
        x_emb = self.scan_blocks_train(self.blocks, (queries, keys, mask))[0]
        return self.output_proj(x_emb)  # Return logits: (B, child len, alphabet_size)
    
    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def _make_conditional_mask(x, parent_pos):
        """Create mask where child positions use causal masking among themselves but can see all parents (keys)."""
        # mask has shape (B, H, L, L); want to omit parents from queries.
        mask = nnx.make_causal_mask(x)[:, :, parent_pos-1:, :] # shape (B, H, child len, L)
        # Allow all positions to attend to parent positions (full bidirectional within parents)
        # mask = mask.at[:, :, :parent_pos].set(True) # this was def wrong dimension; this let parents see everything.
        return mask
    
    def sample_node(self, x, node_index, init_cache=False):
        """Sample a single node from the conditional transformer model.

        Args:
            x: Previous tokens of shape (B, L_current) containing categorical values for child states
            node_index: Index of the node being predicted

        Returns:
            Logits of shape (B, L, n_states) for the next token prediction
        """
        assert node_index < len(self.n_states), "node_index must be less than the number of child states"
        keys = self.pos_embed[None, :x.shape[1], :] + self.token_embed(x) # (B, L_current, D_model)
        query = keys[:, -1, :][:,None,:] # (B, 1, D_model) -- last token of current sequence
        if init_cache:
            # Initialize cache with maximum possible sequence length
            max_seq_len = len(self.n_parent_states_list) + len(self.n_states) - 1 # -1 b/c no next token for last position.
            cache_shape = (keys.shape[0], max_seq_len, keys.shape[2])
            nnx.vmap(
                lambda b: b.attention.init_cache(cache_shape, dtype=jnp.bfloat16)
            )(self.blocks)
        
        for i in range(self.n_layers): # can't scan b/c of init cache (different trace levels)
            query = nnx.vmap(
                lambda b: b(
                    query.astype(jnp.bfloat16),
                    keys.astype(jnp.bfloat16),
                    mask=None,
                    decode=True
                )
            )(self.blocks)[i] # discard all other layers :(
            
        return self.output_proj(query)[
            ..., :self.n_states[node_index] # b/c predicting next token.
        ].squeeze().astype(jnp.float32)  # Return logits: (B, 1, n_states@node_index)
    
    def sample(self, parent_states: jnp.ndarray, rng: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Sample a batch of sequences from the conditional model. 
        Inputs:
            parent_states: conditioning states of shape (n_samples, n_parent_vars)
            rng: random number generator -- shape is n_samples. or None.
        Returns:
            samples: (n_samples, n_nodes)
            log_prob: (n_samples,)
        '''
        n_samples = rng.shape[0] if rng is not None else 1
        samples = jnp.zeros((n_samples, len(self.n_states),), dtype=jnp.int32)
        context = jnp.zeros((n_samples, len(self.mod_states),), dtype=jnp.int32)
        context = context.at[:, :len(self.n_parent_states_list)].set(parent_states)
        log_probs = jnp.zeros((n_samples,))

        for t in range(len(self.n_states)):
            logits = self.sample_node(
                context[:,:len(self.n_parent_states_list)+t], # NOTE: a bit weird to be passing >1 token here.
                t, init_cache=(t==0)
            )  # logits for position t: (B, n_states[t])
            probs = jax.nn.log_softmax(logits) # shape (n_samples, n_states[t])
            if rng is not None:
                rng, srng = jax.vmap(lambda k: jax.random.split(k, 2))(rng).transpose((1, 0, 2))
                tokens = jax.vmap(jax.random.categorical)(srng, probs)
            else:
                tokens = jax.vmap(jnp.argmax)(probs)
            samples = samples.at[:,t].set(tokens) # result
            context = context.at[:,len(self.n_parent_states_list)+t].set(tokens) # next query
            log_probs = log_probs + probs[jnp.arange(n_samples), tokens]

        return samples, log_probs # slice off parent states
    
    def sample_mode(self, parent_states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Sample the mode of the conditional model.
        Args:
            parent_states: conditioning states of shape (1, n_parent_vars) or (n_parent_vars,)
        Returns:
            samples: (n_nodes,)
            log_prob: float
        '''
        if parent_states.ndim == 1:
            parent_states = parent_states[None, :]  # Add batch dimension
        
        samples, log_probs = self.sample(parent_states)
        return samples.squeeze(), log_probs.squeeze()
    
    @staticmethod
    @partial(jax.vmap, in_axes=(0, 1, -1)) # vmap over n_nodes
    def log_probs_node(n_states: int, logits: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        mask = jnp.arange(logits.shape[-1]) < n_states  # (max_states,)
        log_probs = jax.nn.log_softmax(
            jnp.where(mask, logits, -jnp.inf) # invalid states = -inf
        ) # (B, n_states)
        return log_probs[jnp.arange(x.shape[0]), x] # (B,)
    
    def _likelihood(self, x: jnp.ndarray, parent_states: jnp.ndarray) -> jnp.ndarray:
        """Compute likelihood of child states given parent states."""
        keys_queries = jnp.concatenate([
            parent_states, 
            x[:,:-1], # don't need to predict from last token.
        ], axis=1)
        logits = self(keys_queries)  # shape (B, child_len, alphabet_size)
        # NOTE: softmax only over valid alphabet @ each position.
        return self.log_probs_node(jnp.array(self.n_states), logits, x).T / x.shape[-1] # shape (B, L)
    
    def likelihood(self, x: jnp.ndarray, parent_states: jnp.ndarray) -> jnp.ndarray:
        return self._likelihood(x, parent_states).sum(axis=1) # shape (B,)
    
    def fast_likelihood(self, x: jnp.ndarray, parent_states: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment given parent state.
        Non-diff.
        """
        # Convert to pure function to avoid trace level conflicts
        graphdef, state = nnx.split(self)
        
        @jax.jit
        def pure_likelihood(state, x, parent_states):
            keys_queries = jnp.concatenate([
                parent_states, 
                x[:,:-1], # don't need to predict from last token.
            ], axis=1)
            
            # Recreate the model as a pure function
            model = nnx.merge(graphdef, state)
            parent_pos = len(model.n_parent_states_list)
            keys = model.token_embed(keys_queries) + model.pos_embed[None, :, :]  # (B, L, D_model)
            queries = keys[:, parent_pos-1:, :]
            mask = model._make_conditional_mask(keys_queries, parent_pos)
            
            # Use the original scan approach since we're now in a pure function
            x_emb = model.scan_blocks_train(model.blocks, (queries, keys, mask))[0]
            logits = model.output_proj(x_emb)  # Return logits: (B, child len, alphabet_size)
            
            # NOTE: softmax only over valid alphabet @ each position.
            return (model.log_probs_node(jnp.array(model.n_states), logits, x).T / x.shape[-1]).sum(axis=1) # shape (B,)
        
        return pure_likelihood(state, x, parent_states)