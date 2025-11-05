from tkinter import W
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import nnx
import optax as ox
from optax.contrib import muon
from typing import Callable, Tuple, Optional
import time
import matplotlib.pyplot as plt
from src.decomposition.graphs import Tree, JunctionTree

from src.utils import kl_divergence

from src.opt.model_based.EDA.base import EDA
from src.models.density.MLP.joint import JointDensityMLP
from src.models.density.MLP.autoregressive import AutoregressiveDensityMLP
from src.models.density.transformer.autoregressive import AutoregressiveDensityTransformer
# TODO: Add transformer models, VAEs, potts models, etc.

from src.utils import count_params

WEIGHT_CLIP = 1e30 # clip weights to avoid overflow

class JointEDA(EDA):
    """EDA using either joint or autoregressive density models."""
    
    def __init__(self, 
                use_transformer: bool = False,
                *args,
                num_layers: Optional[int] = 4,
                dim_ff: Optional[int] = 64,
                num_heads: Optional[int] = 4,
                dim_attn: Optional[int] = 64,
                replay_buffer: Optional[bool] = False,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.use_transformer = use_transformer
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.num_heads = num_heads
        self.dim_attn = dim_attn
        self.model = self.policy = None
        self.replay_buffer = replay_buffer
        if not use_transformer:
            self.arc_string = "-".join(map(str, [dim_ff]*num_layers))
        else:
            self.arc_string = f"TF-{num_heads}H-{num_layers}L-{dim_attn}A-{dim_ff}FF"

    def precompute(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Train the density model using gradient-based optimization."""
        if self.use_transformer:
            self.policy = AutoregressiveDensityTransformer(
                self.n_states,
                self.dim_attn,
                self.num_heads,
                self.dim_ff,
                self.num_layers,
                rngs=self.rng,
            )
            optimizer = muon # ox.adamw
        else:
            self.policy = AutoregressiveDensityMLP( # can also use JointDensityMLP, but deprecated due to poor scaling.
                self.n_states, # NOTE: should be a list, not jax array
                [self.dim_ff] * self.num_layers,
                rngs=self.rng,
            )
            # self.policy = JointDensityMLP(
            #     self.n_states,
            #     [self.dim_ff] * self.num_layers,
            #     rngs=self.rng,
            # )
            optimizer = ox.adamw # slower w/ muon (tested on tree, joint)
        if self.verbose: print(f"\t{self.policy.__class__.__name__} # params: {count_params(self.policy)}")
        
        if self.use_prior != 'none':
            assert self.prior_data is not None, "Prior data must be provided if using prior."
            assert self.prior_data.ndim == 2, f"Prior data must be 2D (N, L); got {self.prior_data.ndim}D."
            assert self.prior_data.shape[1] == len(self.n_states), f"Prior data must have same number of variables as objective; got {self.prior_data.shape[1]} vs. {len(self.n_states)}."
            self.fit_uniform_prior()
            # maybe needs to be batched? idk...assuming if loss works for policy w/o batching currently, then ok
            prior_likelihood = self.prior.fast_likelihood
            if 'init' in self.use_prior: # use prior as initial policy
                graphdef, state = nnx.split(self.prior)
                self.policy = nnx.merge(graphdef, state)
        else:
            prior_likelihood = lambda x: jnp.zeros(x.shape[0])
        
        nnx_optimizer = nnx.Optimizer(
            self.policy,
            ox.chain(
                optimizer(
                    self.learning_rate,
                )
            )
        )
        
        # Initialize history arrays
        samples_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples, self.n_variables)).astype(jnp.int32)
        values_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples))
        losses_history = jnp.zeros((self.num_epochs + 1,))
        kld_history = jnp.zeros((self.num_epochs + 1,))
        ess_history = jnp.zeros((self.num_epochs + 1,))
        prior_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples))
        weights_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples, 1))
        best_x_frequency = jnp.zeros((self.num_epochs + 1, self.objective.fg.n_nodes))
        
        @jax.jit
        def get_weights(x):
            Q = self.objective.query(x)
            return jnp.clip(jnp.exp(
                Q / self.temp_Q
            ), 0., WEIGHT_CLIP)[:, None]
        
        def loss_fn(policy, samples, values, prior_lps):
            kld = 0. # filler
            log_likelihoods = policy.likelihood(samples).ravel()
            assert log_likelihoods.shape == values.ravel().shape
            wll = values.ravel() * log_likelihoods # (N,)
            ess = (values.sum())**2 / ((values**2).sum() + 1e-8)
            if 'weight' in self.use_prior: # use p0 * exp(Q) instead of just exp(Q). use beta as temperature.
                prior_probs = jnp.exp(prior_lps * self.beta_prior).ravel() # nonneg weights
                return -(wll * prior_probs).mean(), (kld, ess) # could be prior probs + values?
            elif 'kld' in self.use_prior: # mode-seeking direction: penalty if policy has mass outside of prior's support.
                kld = jax.lax.max(log_likelihoods.mean() - prior_lps.mean(), 0.) # TODO: not quite on policy, but close enough?
                return -wll.mean() + self.beta_prior * kld, (kld, ess)
            return -wll.mean(), (kld, ess)
        
        @nnx.jit # notice nnx.jit, not jax.jit
        def update_step(policy, optimizer, samples, values, prior_lps):
            (loss, (kld, ess)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(policy, samples, values, prior_lps)
            optimizer.update(grads)
            return policy, optimizer, loss, kld, ess
        
        self.policy.eval()
        initial_samples, _ = self.policy.sample(
            jax.random.split(self.rng, self.num_MC_samples)
        )
        samples_history = samples_history.at[0].set(initial_samples)
        prior_history = prior_history.at[0].set(prior_likelihood(samples_history[0]))
        values_history = values_history.at[0].set(self.objective.query(samples_history[0]))
        weights_history = weights_history.at[0].set(get_weights(samples_history[0]))
        initial_loss, (initial_kld, initial_ess) = loss_fn(self.policy, samples_history[0], weights_history[0], prior_history[0])
        losses_history = losses_history.at[0].set(initial_loss)
        kld_history = kld_history.at[0].set(initial_kld)
        ess_history = ess_history.at[0].set(initial_ess)

        def inner_step(carry, i):
            policy, optimizer, samples, epoch = carry
            weights = get_weights(samples)
            prior_lps = prior_likelihood(samples)
            policy, optimizer, loss, kld, ess = update_step(policy, optimizer, samples, weights, prior_lps)
            return (policy, optimizer, samples, epoch), (loss, kld, ess)
        if self.replay_buffer:
            # pre-compute replay buffer indices (too messy to do inside scan)
            batch_size = self.num_MC_samples
            rb_inds = jnp.zeros((self.num_epochs + 1, self.num_inner_iter, batch_size), dtype=jnp.int32) # using nmcs as batch size for now
            for epoch in range(self.num_epochs + 1):
                self.rng, sample_rng = jax.random.split(self.rng)
                rb_inds = rb_inds.at[epoch].set(
                    jax.random.choice(sample_rng, (epoch+1)*batch_size, shape=(self.num_inner_iter, batch_size,), replace=True)
                )

            def inner_step(carry, i):
                policy, optimizer, replay_buffer, epoch = carry
                samples = replay_buffer[rb_inds[epoch, i], :] # (B, L)
                weights = get_weights(samples)
                prior_lps = prior_likelihood(samples)
                policy, optimizer, loss, kld, ess = update_step(policy, optimizer, samples, weights, prior_lps)
                return (policy, optimizer, replay_buffer, epoch), (loss, kld, ess)
            
        scan_inner = nnx.scan(
            f=inner_step, length=self.num_inner_iter
        )
        
        # Outer epoch loop
        def epoch_step(carry, epoch):
            policy, optimizer, rng, samples_history, values_history, losses_history, prior_history, kld_history, ess_history, weights_history = carry
            samples = samples_history[epoch]
            if self.replay_buffer:
                samples = samples_history.reshape(-1, self.n_variables)
            
            # Run inner optimization loop
            policy.train()
            init_carry = (policy, optimizer, samples, epoch)
            (policy, optimizer, _, _), (losses, klds, ess) = scan_inner(
                init_carry, jnp.arange(self.num_inner_iter)
            )
            
            # Sample from model for next epoch
            rng, sample_rng = jax.random.split(rng)
            policy.eval()
            samples, _ = policy.sample(
                jax.random.split(sample_rng, self.num_MC_samples)
            )
            
            # Update history arrays functionally
            samples_history = samples_history.at[epoch+1].set(samples)
            values_history = values_history.at[epoch+1].set(self.objective.query(samples))
            weights_history = weights_history.at[epoch+1].set(get_weights(samples))
            losses_history = losses_history.at[epoch+1].set(losses[-1])
            prior_history = prior_history.at[epoch+1].set(prior_likelihood(samples))
            kld_history = kld_history.at[epoch+1].set(klds[-1])
            ess_history = ess_history.at[epoch+1].set(ess[-1])

            return (policy, optimizer, rng, samples_history, values_history, losses_history, prior_history, kld_history, ess_history, weights_history), None
        
        # Run epoch loop
        init_carry = (self.policy, nnx_optimizer,
                      self.rng, samples_history,
                      values_history, losses_history,
                      prior_history, kld_history, ess_history, weights_history
        )
        (self.policy, _, self.rng, samples_history, values_history, losses_history, prior_history, kld_history, ess_history, weights_history), _ = nnx.scan(
            f=epoch_step, length=self.num_epochs
        )(init_carry, jnp.arange(self.num_epochs))
        self.policy.eval()
        self.model = self.policy
        
        results = {
            'samples': samples_history,
            'values': values_history,
            'losses': losses_history,
            'ess': ess_history,
            'Q': values_history[..., None],
            'weights': weights_history,
        }
        if self.use_prior != 'none':
            results['prior'] = prior_history[..., None]
            if 'kld' in self.use_prior:
                results['kld'] = kld_history
            elif 'weight' in self.use_prior:
                results['prior_weights'] = jnp.exp(results['prior'] / self.beta_prior)
                results['weights'] = results['prior_weights'] * results['weights']

        return results