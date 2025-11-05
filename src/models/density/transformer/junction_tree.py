import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Optional, List
from functools import partial

from src.models.blocks import TransformerBlock
from src.models.density.base import DensityModel
from src.models.density.transformer.autoregressive import AutoregressiveDensityTransformer, ConditionalAutoregressiveDensityTransformer
from src.decomposition.graphs import JunctionTree

# NOTE: pays price quadratic in clique sizes instead of full length, but incurs for loop cost (slower training).
class JunctionTreeDensityCliqueTransformers(DensityModel, nnx.Module):
    def __init__(self, junction_tree: JunctionTree, n_states: List[int], d_model: int = 64, n_heads: int = 4, d_ff: int = 128, n_layers: int = 2, *, rngs):
        self.n_states = n_states
        self.junction_tree = junction_tree
        self.root = junction_tree.root
        self.node_order = junction_tree.node_order
        self.parents = junction_tree.parents
        self.children = junction_tree.children
        self.n_nodes = len(self.node_order)
        # TODO: should these actually be learned? I think DADO requires that params are SEPARATE between nodes.
        self.alphabet_size = max(n_states) + 1 # +1 for SOS token
        rngs, token_rng = jax.random.split(rngs, 2)
        self.token_embed_shared = nnx.Embed( # NOTE: requires all models to share d_model (or have addtl shrinkage layer...)
            num_embeddings=self.alphabet_size, features=d_model, rngs=nnx.Rngs(token_rng)
        ) # set attributes in each model.
        
        # Store junction tree structure
        self.index_to_nodes_non_overlapping = [None for _ in range(self.n_nodes)]  # Maps clique index to non-overlapping variable nodes (for model input)
        self.index_to_nodes = [None for _ in range(self.n_nodes)]  # Maps clique index to original clique variables (for parent conditioning)
        
        # Create transformer models for each clique - different sizes for different cliques
        self.models = [None for _ in range(self.n_nodes)]
        model_rngs = jax.random.split(rngs, self.n_nodes)
        for i in range(self.n_nodes):
            clique_vars = sorted(list(self.junction_tree.index_to_nodes[i]))
            self.index_to_nodes[i] = clique_vars
            clique_n_states_list = [n_states[var] for var in clique_vars]
            
            if i == self.root:
                # Root clique uses autoregressive transformer model (no conditioning)
                self.models[i] = AutoregressiveDensityTransformer(
                    clique_n_states_list, d_model, n_heads, d_ff, n_layers, rngs=model_rngs[i]
                )
            else:
                # Non-root cliques use conditional autoregressive transformer model
                parent_idx = self.parents[i]
                parent_clique_vars = sorted(list(self.junction_tree.index_to_nodes[parent_idx]))
                parent_n_states_list = [n_states[var] for var in parent_clique_vars]
                
                # Account for overlap in vars between parent and child cliques
                clique_vars = [var for var in clique_vars if var not in parent_clique_vars]
                clique_n_states_list = [n_states[var] for var in clique_vars]
                
                self.models[i] = ConditionalAutoregressiveDensityTransformer(
                    clique_n_states_list, parent_n_states_list, d_model, n_heads, d_ff, n_layers, rngs=model_rngs[i]
                )
            self.models[i].token_embed = self.token_embed_shared
            self.index_to_nodes_non_overlapping[i] = clique_vars

    @staticmethod
    def _sample_clique(model, parent_states: Optional[jax.Array], rng: jax.random.PRNGKey = None) -> Tuple[jax.Array, float]:
        """Sample from a single clique model."""
        if parent_states is None: # Root clique - use autoregressive transformer
            return model.sample(rng)
        else: # Non-root clique - use conditional autoregressive transformer
            return model.sample(parent_states, rng)

    @staticmethod
    def _sample(models,
                node_order: List[int],
                root: int,
                parents: List[int],
                index_to_nodes: List[set],
                index_to_nodes_non_overlapping: List[set],
                rng: jax.random.PRNGKey) -> Tuple[jax.Array, float]:
        """Sample from the entire junction tree."""
        # Initialize with zeros - will be filled as we sample cliques
        n_original_vars = max(max(clique) for clique in index_to_nodes) + 1
        samples = jnp.zeros((rng.shape[0], n_original_vars,), dtype=jnp.int32)
        total_logp = jnp.zeros((rng.shape[0],))
        
        for clique_idx in node_order:
            clique_vars = index_to_nodes_non_overlapping[clique_idx] # non-overlapping vars for model input
            # Determine parent state for conditioning
            if clique_idx == root:
                parent_states = None
            else:
                parent_idx = parents[clique_idx]
                parent_clique_vars = index_to_nodes[parent_idx]
                # Extract parent clique assignment from current samples
                parent_states = samples[:, jnp.array(parent_clique_vars)]
            
            # Sample from this clique
            rng, clique_rng = jax.vmap(lambda k: jax.random.split(k, 2))(rng).transpose((1, 0, 2))
            clique_assignment, clique_logp = JunctionTreeDensityCliqueTransformers._sample_clique(
                models[clique_idx], parent_states, rng=clique_rng
            )
            
            # Update the overall assignment with this clique's variables
            for i, var in enumerate(clique_vars):
                samples = samples.at[:,var].set(clique_assignment[:, i])
            
            total_logp = total_logp + clique_logp
        
        return samples, total_logp

    def sample(self, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, float]:
        """Sample a complete assignment from the junction tree model."""
        result = self._sample(
            self.models,
            self.node_order,
            self.root,
            self.parents,
            self.index_to_nodes,
            self.index_to_nodes_non_overlapping,
            rng=rng, # should already be [n_samples]
        )
        return result
    
    def sample_mode(self) -> Tuple[jnp.ndarray, float]:
        """Sample the mode of the junction tree model."""
        self.eval()
        result = self._sample(
            self.models,
            self.node_order,
            self.root,
            self.parents,
            self.index_to_nodes,
            self.index_to_nodes_non_overlapping,
            rng=None,
        )
        self.train()
        return result[0].squeeze(), result[1].squeeze()
    
    def sample_original(self, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, float]:
        """Sample and return in original variable order (same as sample for this implementation)."""
        return self.sample(rng)

    def _likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment."""
        total_logp = jnp.zeros((x.shape[0], x.shape[-1])) # shape (n_samples, L)
        
        for clique_idx in self.node_order: # NOTE: can't vmap b/c separate models!
            clique_vars_non_overlapping = jnp.array(self.index_to_nodes_non_overlapping[clique_idx]) # non-overlapping vars for model input
            
            if clique_idx == self.root:
                # Root clique
                logp = self.models[clique_idx]._likelihood(x[:, clique_vars_non_overlapping]) # can use either here.
            else:
                # Non-root clique - need parent states (individual states, not mixed-radix)
                parent_idx = self.parents[clique_idx]
                parent_clique_vars = self.index_to_nodes[parent_idx]  # original parent vars
                parent_states = x[:, jnp.array(parent_clique_vars)]
                
                logp = self.models[clique_idx]._likelihood(x[:, clique_vars_non_overlapping], parent_states)
            
            total_logp = total_logp.at[:, clique_vars_non_overlapping].set(logp)
        
        return total_logp # pre-normalized by length
    
    def likelihood(self, x: jnp.ndarray) -> float:
        return self._likelihood(x).sum(axis=1) # pre-normalized by length
    
    def clique_likelihood(
        self,
        x: jnp.ndarray,
    ) -> float:
        """Compute the weighted log-likelihood for junction tree using sample-based Q values."""
        logps = jnp.zeros((x.shape[0], len(self.node_order))) # shape (n_samples, L) in JT vars.
        
        for clique_idx in self.node_order:
            clique_vars_non_overlapping = jnp.array(self.index_to_nodes_non_overlapping[clique_idx]) # non-overlapping vars for model input
            
            # Compute likelihood for this clique
            if clique_idx == self.root:
                logp = self.models[clique_idx]._likelihood(x[:, clique_vars_non_overlapping]) # can use either here.
            else:
                parent_idx = self.parents[clique_idx]
                parent_clique_vars = self.index_to_nodes[parent_idx]  # original parent vars
                parent_states = x[:, jnp.array(parent_clique_vars)]
                logp = self.models[clique_idx]._likelihood(x[:, clique_vars_non_overlapping], parent_states)
            
            logps = logps.at[:,clique_idx].set(logp.sum(axis=-1))
        
        return logps
    
    def clique_weighted_likelihood(
        self,
        x: jnp.ndarray,
        Q: jnp.ndarray, # Q[n_cliques] - Q values for this sample
    ) -> float:
        """Compute the weighted log-likelihood for junction tree using sample-based Q values."""
        logps = self.clique_likelihood(x) # shape (n_samples, n_cliques)
        weights = Q # shape (n_samples, n_cliques)
        
        return (logps * weights).sum(axis=-1) # pre-normalized by length

    def _clique_weighted_likelihood(
        self,
        x: jnp.ndarray,
        Q: jnp.ndarray, # Q[n_cliques] - Q values for this sample
    ) -> float:
        """Compute the weighted log-likelihood for junction tree using sample-based Q values."""
        logps = self.clique_likelihood(x) # shape (n_samples, n_cliques)
        weights = Q # shape (n_samples, n_cliques)
        
        return logps, weights # pre-normalized by length

    def fast_likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment using fast_likelihood methods.
        Non-diff.
        """
        total_logp = jnp.zeros((x.shape[0],))
        
        for clique_idx in self.node_order:
            clique_vars_non_overlapping = jnp.array(self.index_to_nodes_non_overlapping[clique_idx])
            
            if clique_idx == self.root:
                # Root clique
                logp = self.models[clique_idx].fast_likelihood(x[:, clique_vars_non_overlapping])
            else:
                # Non-root clique - need parent states
                parent_idx = self.parents[clique_idx]
                parent_clique_vars = self.index_to_nodes[parent_idx]
                parent_states = x[:, jnp.array(parent_clique_vars)]
                
                logp = self.models[clique_idx].fast_likelihood(x[:, clique_vars_non_overlapping], parent_states)
            
            total_logp = total_logp + logp
        
        return total_logp


# NOTE: Single transformer is faster for training but pays quadratic memory cost for using full length.
class JunctionTreeDensityTransformer(DensityModel, nnx.Module):
    def __init__(self, junction_tree: JunctionTree, n_states: List[int], d_model: int = 64, n_heads: int = 4, d_ff: int = 128, n_layers: int = 2, *, rngs):
        self.n_states = n_states
        self.junction_tree = junction_tree
        self.root = junction_tree.root
        self.parents = junction_tree.parents
        self.children = junction_tree.children
        self.clique_order = tuple(junction_tree.node_order)
        self.n_cliques = len(self.clique_order)
        
        # Store junction tree structure as tuples for JAX static args
        clique_to_nodes = [None]*len(self.clique_order)
        clique_to_parents = [None]*len(self.clique_order)
        node_to_clique = [None]*len(n_states)
        
        for clique_idx in self.clique_order:
            clique_vars = tuple(sorted(list(self.junction_tree.index_to_nodes[clique_idx])))
            
            if clique_idx == self.root:
                clique_to_parents[clique_idx] = ()  # Root has no parents
            else:
                parent_idx = junction_tree.parents[clique_idx]
                parent_vars = tuple(sorted(list(self.junction_tree.index_to_nodes[parent_idx])))
                clique_to_parents[clique_idx] = parent_vars
                clique_vars = tuple(var for var in clique_vars if var not in parent_vars) # non-overlapping vars for model input
            clique_to_nodes[clique_idx] = clique_vars
            for var in clique_vars: # reverse mapping
                node_to_clique[var] = clique_idx
        self.clique_to_nodes = tuple(clique_to_nodes)  # Tuple of tuples
        self.clique_to_parents = tuple(clique_to_parents)  # Tuple of tuples
        self.node_to_clique = tuple(node_to_clique)  # Tuple of ints
        
        # Create function for log probs computation that handles variable-length cliques
        def _compute_all_clique_logps(node_logps):
            clique_logps = [None] * len(self.clique_order)
            for clique_idx in self.clique_order:
                clique_logps[clique_idx] = node_logps[:, jnp.array(self.clique_to_nodes[clique_idx])].sum(axis=-1)
            return jnp.stack(clique_logps, axis=0)  # shape (n_cliques, n_samples)
        
        self.log_probs_clique = jax.jit(_compute_all_clique_logps)
        
        # Single transformer setup (like tree transformer but for junction tree)
        self.SOS_TOKEN_ID = max(n_states)
        self.mod_states = [1] + [ns + 1 for ns in n_states]  # +1 for SOS token
        self.alphabet_size = max(self.mod_states)
        self.n_layers = n_layers
        
        rngs, *model_rngs = jax.random.split(rngs, 1 + 3)
        self.token_embed = nnx.Embed(
            num_embeddings=self.alphabet_size, features=d_model, rngs=nnx.Rngs(model_rngs[0])
        )
        self.pos_embed = nnx.Param(
            jax.random.normal(model_rngs[1], (len(n_states), d_model))  # Position for each variable
        )
        self.output_proj = nnx.Linear(d_model, self.alphabet_size, rngs=nnx.Rngs(model_rngs[-1]))
        
        layer_rngs = jax.random.split(rngs, n_layers)
        @nnx.vmap(
            out_axes=nnx.StateAxes({
                nnx.Param: 0,
                nnx.RngState: 0,
                ...: None,
            })
        )
        def create_block(key):
            return TransformerBlock(d_model, n_heads, d_ff, rngs=nnx.Rngs(key))
        self.blocks = create_block(layer_rngs)
        
        # Build parent queries for junction tree structure (similar to tree transformer)
        self.parent_queries = self.get_parent_queries()

    def get_parent_queries(self) -> jnp.ndarray:
        """Build parent queries for each variable position similar to tree transformer.
        These will index into x_with_sos in node_likelihood.
        Parent should now be the last variable sampled from the parent clique or the
        last variable sampled from own clique, or SOS. """
        parent_inds = jnp.zeros((len(self.n_states),), dtype=jnp.int32)
        prev_clique_var = -1
        for clique_idx in self.clique_order:
            clique_vars = self.clique_to_nodes[clique_idx]
            prev_var = -1
            for var_idx in clique_vars: # sorted
                if prev_var == -1: # first var in clique
                    if clique_idx == self.root:
                        assert prev_clique_var == -1, "Root clique should have no parent"
                        parent_inds = parent_inds.at[var_idx].set(0) # SOS
                    else:
                        assert prev_clique_var != -1, "Non-root clique should have parent"
                        parent_inds = parent_inds.at[var_idx].set(prev_clique_var+1) # shift by SOS token.
                else: # other vars in clique just use previous var. -- AR within clique.
                    parent_inds = parent_inds.at[var_idx].set(prev_var+1) # shift by SOS token.
                prev_var = var_idx # for next var in clique.
                prev_clique_var = prev_var # so will be final var in clique for next clique.
        return parent_inds

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
            x: Query tokens of shape (B, L) containing categorical values (SOS/parents of each position)

        Returns:
            Logits of shape (B, L, alphabet_size) for **current** token prediction
        """
        x_emb = self.token_embed(x) + self.pos_embed[None, :, :]  # (B, L, D_model)
        mask = self._make_junction_tree_mask(x, self.clique_to_nodes, self.clique_to_parents, self.node_to_clique)
        x_emb = self.scan_blocks_train(self.blocks, (x_emb, mask))[0]
        return self.output_proj(x_emb)  # Return logits: (B, L, alphabet_size)

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2, 3))
    def _make_junction_tree_mask(x, clique_to_nodes, clique_to_parents, node_to_clique):
        """Create mask based on junction tree structure - variables can only attend to their
        parent clique vars and previous vars in own clique."""
        batch_size, seq_len = x.shape
        mask = jnp.zeros((batch_size, seq_len, seq_len), dtype=jnp.bool_)
        
        for i in range(seq_len):
            # Self-attention; this covers SOS token case for root clique first var.
            mask = mask.at[:, i, i].set(True)

            clique_idx = node_to_clique[i]
            parent_clique_vars = clique_to_parents[clique_idx]
            # attend to all vars in parent clique
            mask = mask.at[
                :, i, 
                jnp.array(parent_clique_vars).astype(int), # have to cast to int b/c jnp.array(tuple) --> float default.
            ].set(True)
            # attend to all previous vars in own clique (fully AR within clique)
            for var in clique_to_nodes[clique_idx]:
                if var < i: # can use b/c sorted.
                    mask = mask.at[:, i, var].set(True)
        
        return mask[:, None, :, :]  # Add singleton dim. for MHA: (B, 1, L, L)

    def sample_node(self, x, node_index, init_cache=False):
        """Sample a single node from the transformer model.

        Args:
            x: Previous tokens of shape (B, L_current) containing categorical values  
            node_index: Index of the node being predicted (0-based)

        Returns:
            Logits of shape (B, n_states) for the token prediction
        """
        assert node_index < len(self.n_states), "node_index must be less than the number of nodes"
        x_emb = self.pos_embed[None, node_index, :] + self.token_embed(x)  # (B, 1, D_model)
        
        if init_cache:
            # Initialize cache with maximum possible sequence length
            max_seq_len = len(self.n_states)
            cache_shape = (x_emb.shape[0], max_seq_len, x_emb.shape[2])
            nnx.vmap(
                lambda b: b.attention.init_cache(cache_shape, dtype=jnp.bfloat16)
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
        ].squeeze().astype(jnp.float32)  # Return logits: (B, n_states@node)

    def sample(self, rng: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Sample a batch of sequences from the model in junction tree order. 
        Inputs:
            rng: random number generator -- shape is n_samples. or None.
        Returns:
            samples: (n_samples, n_nodes) 
            log_prob: (n_samples,)
        '''
        n_samples = rng.shape[0] if rng is not None else 1
        samples = jnp.zeros((n_samples, len(self.n_states),), dtype=jnp.int32)
        log_probs = jnp.zeros((n_samples,))
        
        # Sample variables in the order they appear in the junction tree decomposition
        for var_idx in range(len(self.n_states)): # NOTE: can't scan b/c of init cache (different trace levels + cond.) maybe could do after first sample?
            # Get parent token based on parent queries
            parent_token_idx = self.parent_queries[var_idx]
            parent_token = jax.lax.select(
                parent_token_idx == 0, # SOS token
                jnp.full((n_samples,), self.SOS_TOKEN_ID, dtype=jnp.int32),
                samples[:, parent_token_idx - 1]  # -1 because parent_queries accounts for SOS
            )
            
            logits = self.sample_node(parent_token[:, None], var_idx, init_cache=(var_idx == 0))
            probs = jax.nn.log_softmax(logits) # shape (n_samples, n_states[var_idx])
            
            if rng is not None:
                rng, srng = jax.vmap(lambda k: jax.random.split(k, 2))(rng).transpose((1, 0, 2))
                tokens = jax.vmap(jax.random.categorical)(srng, probs).squeeze()
            else:
                tokens = jax.vmap(jnp.argmax)(probs).squeeze()
            
            samples = samples.at[:, var_idx].set(tokens)
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
    
    @staticmethod
    @partial(jax.vmap, in_axes=(0, 1, -1)) # vmap over n_nodes
    def log_probs_node(n_states: int, logits: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        mask = jnp.arange(logits.shape[-1]) < n_states  # (max_states,)
        log_probs = jax.nn.log_softmax(
            jnp.where(mask, logits, -jnp.inf) # invalid states = -inf
        ) # (B, n_states)
        return log_probs[jnp.arange(x.shape[0]), x] # (B,)
    
    def node_likelihood(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-likelihood for each node (position). This can be aggregated to get clique likelihoods."""
        x_with_sos = jnp.concatenate([
            jnp.full(shape=[x.shape[0], 1], fill_value=self.SOS_TOKEN_ID, dtype=jnp.int32), 
            x
        ], axis=1) # shape (B, L+1)
        queries = x_with_sos[:, self.parent_queries] # rearrange s.t. each position has its parent token. shape (B, L)
        logits = self(queries)  # shape (B, L, alphabet_size)
        return self.log_probs_node(jnp.array(self.n_states), logits, x).T / x.shape[-1] # shape (B, L)
    
    def clique_likelihood(
        self,
        x: jnp.ndarray,
    ) -> float:
        """Compute the log-likelihood for each clique of the junction tree."""
        node_logps = self.node_likelihood(x) # shape (B, L)
        
        # Use precomputed jitted function
        clique_log_probs = self.log_probs_clique(node_logps) # shape (n_cliques, n_samples)
        
        return clique_log_probs.T # shape (n_samples, n_cliques), i = clique_idx, not clique_order[i].
    
    def clique_weighted_likelihood(
        self,
        x: jnp.ndarray,
        Q: jnp.ndarray, # Q[n_cliques] - Q values for this sample
    ) -> float:
        """Compute the weighted log-likelihood for junction tree using sample-based Q values."""
        clique_log_probs = self.clique_likelihood(x) # shape (n_samples, n_cliques)
        # NOTE: indices need to be consistent w/ clique_likelihood order.
        weights = Q # shape (n_samples, n_cliques)
        
        return (clique_log_probs * weights).sum(axis=-1) # pre-normalized by length

    def _clique_weighted_likelihood(
        self,
        x: jnp.ndarray,
        Q: jnp.ndarray, # Q[n_cliques] - Q values for this sample
    ) -> float:
        """Compute the weighted log-likelihood for junction tree using sample-based Q values."""
        clique_log_probs = self.clique_likelihood(x) # shape (n_samples, n_cliques)
        # NOTE: indices need to be consistent w/ clique_likelihood order.
        weights = Q # shape (n_samples, n_cliques) # this was previously indexed by clique_order; I think wrong...
        
        return clique_log_probs, weights # pre-normalized by length
    
    def _likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment."""
        return self.clique_likelihood(x)
    
    def likelihood(self, x: jnp.ndarray) -> float:
        return self.node_likelihood(x).sum(axis=1) # pre-normalized by length

    def fast_likelihood(self, x: jnp.ndarray) -> float:
        """Compute the log-likelihood of a complete assignment.
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
            queries = x_with_sos[:, model.parent_queries] # rearrange s.t. each position has its parent token. shape (B, L)
            
            x_emb = model.token_embed(queries) + model.pos_embed[None, :, :]  # (B, L, D_model)
            mask = model._make_junction_tree_mask(queries, model.clique_to_nodes, model.clique_to_parents, model.node_to_clique)
            
            # Use the original scan approach since we're now in a pure function
            x_emb = model.scan_blocks_train(model.blocks, (x_emb, mask))[0]
            logits = model.output_proj(x_emb)  # Return logits: (B, L, alphabet_size)
            
            # NOTE: softmax only over valid alphabet @ each position.
            node_logps = model.log_probs_node(jnp.array(model.n_states), logits, x).T / x.shape[-1] # shape (B, L)
            return model.log_probs_clique(node_logps).T # shape (n_samples, n_cliques)

        return pure_likelihood(state, x)