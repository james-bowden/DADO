import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Optional, List
from functools import partial
from warnings import warn

from src.models.density.transformer.autoregressive import AutoregressiveDensityTransformer
from src.models.density.base import DensityModel

class IndependentDensityTransformer(DensityModel, nnx.Module):
    def __init__(self, n_states: List[int], d_model: int = 64, n_heads: int = 4, d_ff: int = 128, n_layers: int = 2, *, rngs):
        warn("You should probably use IndependentDensityMLP instead.")

        self.n_states = n_states
        self.SOS_TOKEN_ID = 1 + max(n_states)  # start-of-sequence token. Allocate extra length, state below.

        # TODO: make this one transformer w/ particular masking?
        rngs, *model_rngs = jax.random.split(rngs, 1 + n_layers)
        self.models = [ # TODO: vmap these? only works if D is same.
            AutoregressiveDensityTransformer(
                n_states=[1, D+1],
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                n_layers=n_layers,
                rngs=nnx.Rngs(model_rngs[i])
            ) for i, D in enumerate(n_states)
        ]

    def sample(self, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Sample a batch of sequences from the model. 
        Inputs:
            rng: random number generator -- shape is n_samples.
        Returns:
            samples: (n_samples, n_nodes)
            log_prob: (n_samples,)
        '''
        n_samples = rng.shape[0]
        samples = jnp.zeros((n_samples, len(self.n_states),), dtype=jnp.int32)
        log_probs = jnp.zeros((n_samples,))

        for t in range(len(self.n_states)):
            s, lp = self.models[t].sample(rng)
            samples = samples.at[:,t].set(s)
            log_probs = log_probs + lp
        return samples, log_probs
    
    def sample_mode(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Sample the mode of the model.
        Returns:
            samples: (n_nodes,)
            log_prob: float
        '''
        samples = jnp.zeros((1, len(self.n_states),), dtype=jnp.int32)
        log_probs = jnp.zeros((1,))

        for t in range(len(self.n_states)):
            s, lp = self.models[t].sample_mode()
            samples = samples.at[:,t].set(s)
            log_probs = log_probs + lp
        return samples.squeeze(), log_probs.squeeze()
    
    def _likelihood(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.node_likelihood(x)
    
    def likelihood(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.node_likelihood(x).mean(axis=1) # mean likelihood

    def node_likelihood(self, x: jnp.ndarray) -> jnp.ndarray: # to be used for e.g., prior, where no training happening.
        log_probs = jnp.zeros(x.shape[0])
        for t in range(len(self.n_states)):
            log_probs = log_probs.at[t].set(self.models[t].node_likelihood(x[:,t]))
        return log_probs