import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import nnx
import optax as ox
from typing import Callable, Tuple
from functools import partial

from src.opt.model_based.EDA.DADO.base import DADO
from src.models.density.MLP.independent import IndependentDensityMLP

from src.utils import count_params

LOSSES_FULL = False
WEIGHT_CLIP = 1e30 # clip weights to avoid overflow

class IndependentDADO(DADO):
    """EDA using function decomposition and tree-structured density models."""
    
    def __init__(self, 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert self.fg.graph.number_of_edges() == 0, "IndependentDADO requires a FunctionalGraph with no edges."

    # TODO: handle auxiliary variables. 
    # may need to have generalized graph just do all this code in same loop... or extract the sampling parts.
    def precompute(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Train the density model using gradient-based optimization."""
        self.policy = IndependentDensityMLP(
            self.n_states,
            self.hidden_dims,
            rngs=self.rng,
        )
        if self.verbose: print(f"\t{self.policy.__class__.__name__} # params: {count_params(self.policy)}")
        optimizer = ox.adamw(self.learning_rate)
        nnx_optimizer = nnx.Optimizer(self.policy, optimizer)
        
        # Initialize history arrays
        samples_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples, self.n_variables)).astype(jnp.int32)
        values_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples))
        if LOSSES_FULL:
            losses_history = jnp.zeros((self.num_epochs + 1, self.num_inner_iter))
            ess_history = jnp.zeros((self.num_epochs + 1, self.num_inner_iter, self.n_variables))
        else:
            losses_history = jnp.zeros((self.num_epochs + 1,))
            ess_history = jnp.zeros((self.num_epochs + 1, self.n_variables))
        Q_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples, self.fg.n_nodes))
        weights_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples, self.fg.n_nodes))

        # "Compute" value functions -- no edges.
        V = jnp.array(self.fg.node_functions) # [n_nodes, n_states]

        @partial(jax.vmap, in_axes=(None, 0))
        def get_V_weights(V, x):
            return V[jnp.arange(self.fg.n_nodes), x] # (n_nodes,)
        
        @jax.jit
        def std_exp_weights(V):
            return jnp.clip(jnp.exp(
                V / self.temp_Q
            ), 0., WEIGHT_CLIP)

        def loss_fn(policy, samples, V_weights):
            node_lps = policy.node_likelihood(samples) # (N, L)
            ess = (V_weights.sum(axis=0))**2 / ((V_weights**2).sum(axis=0) + 1e-8) # (L,)
            return -jax.vmap(jnp.dot)(node_lps, V_weights).mean(), ess
            
        @nnx.jit # notice nnx.jit, not jax.jit
        def update_step(policy, optimizer, samples, values):
            (loss, ess), grads = nnx.value_and_grad(loss_fn, has_aux=True)(policy, samples, values)
            optimizer.update(grads)
            return policy, optimizer, loss, ess
        
        self.policy.eval()
        initial_samples, _ = self.policy.sample(
            jax.random.split(self.rng, self.num_MC_samples)
        )
        samples_history = samples_history.at[0].set(initial_samples)
        values_history = values_history.at[0].set(self.fg(samples_history[0]))
        Q_history = Q_history.at[0].set(get_V_weights(V, samples_history[0]))
        initial_loss, initial_ess = loss_fn(self.policy, samples_history[0], Q_history[0])
        losses_history = losses_history.at[0].set(initial_loss)
        ess_history = ess_history.at[0].set(initial_ess)
        weights_history = weights_history.at[0].set(std_exp_weights(Q_history[0]))
        
        # Inner optimization loop
        def inner_step(carry, _):
            policy, optimizer, samples, values = carry
            policy, optimizer, loss, ess = update_step(policy, optimizer, samples, values)
            return (policy, optimizer, samples, values), (loss, ess)
        
        scan_inner = nnx.scan(
            f=inner_step, length=self.num_inner_iter
        )
        
        # Outer epoch loop
        def epoch_step(carry, epoch):
            policy, optimizer, V, rng, samples_history, values_history, losses_history, Q_history, ess_history, weights_history = carry
            samples = samples_history[epoch]
            V_weights = weights_history[epoch]
            
            # Run inner optimization loop
            policy.train()
            init_carry = (policy, optimizer, samples, V_weights)
            (policy, optimizer, _, _), (losses, ess) = scan_inner(
                init_carry, jnp.arange(self.num_inner_iter)
            )
            if LOSSES_FULL:
                current_loss = losses
                current_ess = ess
            else: # just store last loss
                current_loss = losses[-1]
                current_ess = ess[-1]
            
            # Sample from model for next epoch
            rng, sample_rng = jax.random.split(rng)
            policy.eval()
            samples, _ = policy.sample(
                jax.random.split(sample_rng, self.num_MC_samples)
            )
            
            samples_history = samples_history.at[epoch+1].set(samples)
            values_history = values_history.at[epoch+1].set(self.fg(samples))
            losses_history = losses_history.at[epoch+1].set(current_loss)
            Q_history = Q_history.at[epoch+1].set(get_V_weights(V, samples))
            weights_history = weights_history.at[epoch+1].set(std_exp_weights(Q_history[epoch+1]))
            ess_history = ess_history.at[epoch+1].set(current_ess)
            
            return (policy, optimizer, V, rng, samples_history, values_history, losses_history, Q_history, ess_history, weights_history), None
        
        # Run epoch loop
        init_carry = (self.policy, nnx_optimizer, V, self.rng, samples_history, values_history, losses_history, Q_history, ess_history, weights_history)
        (self.policy, _, _, self.rng, samples_history, values_history, losses_history, Q_history, ess_history, weights_history), _ = nnx.scan(
            f=epoch_step, length=self.num_epochs
        )(init_carry, jnp.arange(self.num_epochs))
        
        self.policy.eval()
        self.model = self.policy
    
        results = {
            'samples': samples_history,
            'values': values_history,
            'losses': losses_history,
            'ess': ess_history,
            'Q': Q_history,
            'weights': weights_history,
        }
        # if self.use_prior != 'none':
        #     results['prior'] = prior_history[..., None]
        #     if 'kld' in self.use_prior:
        #         results['kld'] = kld_history
        #     elif 'weight' in self.use_prior:
        #         results['prior_weights'] = jnp.exp(results['prior'] / self.beta_prior)
        #         results['weights'] = results['prior_weights'] * results['weights']

        return results
    