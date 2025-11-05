import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax as ox
from optax.contrib import muon
from typing import Callable, Tuple
import networkx as nx
from functools import partial
import matplotlib.pyplot as plt

from src.opt.model_based.EDA.DADO.base import DADO
from src.models.density.MLP.tree import TreeDensityMLP
from src.models.density.transformer.tree import TreeDensityTransformer
# from src.models.regression.MLP import IndependentRegressionMLP, JointRegressionMLP
from src.decomposition.graphs import Tree, JunctionTree

from src.utils import count_params

WEIGHT_CLIP = 1e30 # clip weights to avoid overflow

class TreeDADO(DADO):
    """EDA using function decomposition and tree-structured density models."""
    
    def __init__(self, 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.fg, Tree), "TreeDADO requires a FunctionalTree."
    
    def precompute(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Train the density model using gradient-based optimization."""
        if self.use_transformer:
            self.policy = TreeDensityTransformer(
                self.fg,
                self.n_states,
                self.dim_attn,
                self.num_heads,
                self.dim_ff,
                self.num_layers,
                rngs=self.rng,
            )
            optimizer = muon # ox.adamw
        else:
            self.policy = TreeDensityMLP(
                self.fg,
                self.n_states,
                self.hidden_dims,
                rngs=self.rng,
            )
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
            prior_likelihood = lambda x: jnp.zeros((x.shape[0], len(self.fg.node_order)))
            
        nnx_optimizer = nnx.Optimizer(
            self.policy,
            ox.chain(
                optimizer(
                    self.learning_rate,
                )
            )
        )
        
        samples_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples, self.n_variables)).astype(jnp.int32)
        values_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples))
        losses_history = jnp.zeros((self.num_epochs + 1,))
        kld_history = jnp.zeros((self.num_epochs + 1,))
        ess_history = jnp.zeros((self.num_epochs + 1, self.fg.n_nodes))
        Q_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples, self.fg.n_nodes))
        prior_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples, self.fg.n_nodes))
        weights_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples, self.fg.n_nodes))
        
        # Convert lists to tuples for JAX static arguments 
        children_tuple = tuple(tuple(child_list) for child_list in self.fg.children)
        parents_tuple = tuple(self.fg.parents)
        node_order_tuple = tuple(self.fg.node_order)
        
        # Define value function computation based on previous model.
        @partial(jax.jit, static_argnames=['n_states_list', 'node_order', 'parents', 'children', 'root'])
        def compute_value_functions(samples, n_states_list, V, Q, edge_functions, node_order, parents, children, root):
            # V[node, x_node]: [n_nodes, max_n_states] (padded for different state sizes)
            # Q[node, x_parent, x_node]: [n_nodes, max_n_states, max_n_states] (padded)
            # Conceptually, it's Q[(parent, node), x_parent, x_node] -- defined on edge. 
            # Can define this way b/c tree => each node has only one parent.

            # NOTE: we start w/ V = f(x_node)
            for node in node_order[::-1]: # Process each node from leaves --> root
                node_n_states = n_states_list[node]
                
                # NOTE: receive messages from children (sample-based)
                # V(x_node) += \sum_{x_child \in children(x_node)} of ...
                # E_{j \sim p(x_child|x_node)} Q(x_node, j)
                for child in children[node]:
                    def compute_update_for_parent_state(p_state):
                        # Create a mask instead of filtering --> "weighted" avg
                        relevant = samples[:, node] == p_state
                        child_vals = samples[:, child]
                        Q_vals = Q[child, jnp.full(samples.shape[0], p_state), child_vals]
                        # Safe division that returns 0 if no relevant samples
                        Q_relevant_mean = jax.lax.select(
                            relevant.sum() > 0,
                            (Q_vals * relevant).sum() / relevant.sum(),
                            0.
                        )
                        return Q_relevant_mean
                    
                    # Vectorize over all parent states for this node
                    updates = jax.vmap(compute_update_for_parent_state)(jnp.arange(node_n_states))
                    V = V.at[node, :node_n_states].add(updates)

                # NOTE: pass messages up to parent
                # Q(node, x_parent, x_node) = f(x_parent, x_node) + V(x_node)
                p = parents[node]
                if p != -1: # NOTE: skip root node
                    parent_n_states = n_states_list[p]
                    f_p_n = edge_functions[(p, node)] # [parent_n_states, node_n_states]
                    
                    def compute_Q_update_for_parent_state(p_state):
                        return V[node, :node_n_states] + f_p_n[p_state, :node_n_states]
                    
                    # Vectorize over all parent states
                    updates = jax.vmap(compute_Q_update_for_parent_state)(jnp.arange(parent_n_states))
                    Q = Q.at[node, :parent_n_states, :node_n_states].set(updates)

            root_n_states = n_states_list[root]
            # Q = Q.at[root, :, :root_n_states].set(V[root, None, :root_n_states])

            # Make advantage instead. How much better is pos than avg decision from parent?
            # NOTE: b/c there are multiple children, advantage is not so straightforward.
            # Really, want Q(clique, x_parent_clique, x_clique) - E_xclique [Q(clique, x_parent_clique, x_clique)]
            A = jnp.zeros_like(Q)
            A = A.at[root, :, :root_n_states].set(V[root, None, :root_n_states])
            for child in range(len(node_order)):
                parent = parents[child]
                if parent != -1:
                    def compute_expected_Q(p_state):
                        # Create a mask instead of filtering --> "weighted" avg
                        relevant = samples[:, parent] == p_state
                        child_vals = samples[:, child]
                        Q_vals = Q[child, jnp.full(samples.shape[0], p_state), child_vals]
                        # Safe division that returns 0 if no relevant samples
                        Q_relevant_mean = jax.lax.select(
                            relevant.sum() > 0,
                            (Q_vals * relevant).sum() / relevant.sum(),
                            0.
                        )
                        return Q_relevant_mean
                    
                    # Vectorize over all parent states for this node
                    V_parent = jax.vmap(compute_expected_Q)(jnp.arange(n_states_list[parent])) # (n_states_parent,)
                    A = A.at[child, ...].set( # (n_states_parent, n_states_pos)
                        Q[child, ...] - V_parent[..., None]) # (n_states_parent, 1)
            return A
        
        # Create appropriately sized V and Q arrays (padded to max states)
        max_n_states = max(self.n_states)
        V_init = jnp.zeros((self.fg.n_nodes, max_n_states))

        # TODO: use generic node function / edge function from tabular or mlp FGM.
        
        # Initialize V with node functions
        for node in range(self.fg.n_nodes):
            node_n_states = self.n_states[node]
            V_init = V_init.at[node, :node_n_states].set(self.fg.node_functions[node][:node_n_states])
        
        # NOTE: avoid having to pass tons of stuff to nnx.scan later.
        compute_value_functions_jit = jax.jit(
            lambda samples: 
            compute_value_functions(
                samples, tuple(self.n_states), 
                V_init,
                jnp.zeros((self.fg.n_nodes, max_n_states, max_n_states)), 
                self.fg.edge_functions, # don't pass node_functions in b/c we'll init V w/ them.
                node_order_tuple, 
                parents_tuple, children_tuple,
                self.fg.root,
            )
        )

        @partial(jax.vmap, in_axes=(None, 0))
        def pick_weights(Q, x):
            parent_indices = jnp.array(self.fg.parents).at[self.fg.root].set(0)
            weights = Q[jnp.arange(self.fg.n_nodes), x[parent_indices], x] # (n_nodes,)
            return weights
        
        @jax.jit
        def get_weights(x):
            Q = compute_value_functions_jit(x)
            return jnp.clip(jnp.exp(
                # ((Q - Q.mean(axis=0)) / (Q.std(axis=0)[None, :] + 1e-8)) / self.temp_Q
                pick_weights(Q, x) / self.temp_Q
            ), 0., WEIGHT_CLIP)
        
        # change to node_advantage_weighted_likelihood if use_advantage
        def loss_fn(policy, samples, Q_weights, prior_lps):
            kld = 0. # filler
            node_lps = policy.node_likelihood(samples) # (N, L)
            ess = (Q_weights.sum(axis=0))**2 / ((Q_weights**2).sum(axis=0) + 1e-8) # (N, L) --> (L,)
            assert node_lps.shape == Q_weights.shape
            wll = (Q_weights * node_lps)
            if 'weight' in self.use_prior: # use p0 * exp(Q) instead of just exp(Q). use beta as temperature.
                prior_probs = jnp.exp(prior_lps * self.beta_prior) # nonneg weights
                assert prior_probs.shape == Q_weights.shape
                return -(jax.vmap(jnp.dot)(Q_weights * prior_probs, node_lps)).mean(), (kld, ess)
            elif 'kld' in self.use_prior: # mode-seeking direction: penalty if policy has mass outside of prior's support.
                kld = jax.lax.max((node_lps.sum(axis=-1) - prior_lps.sum(axis=-1)).mean(), 0.) # TODO: not quite on policy, but close enough?
                return -(wll.sum(axis=-1)).mean() + self.beta_prior * kld, (kld, ess)
            return -(wll.sum(axis=-1)).mean(), (kld, ess)
        
        @nnx.jit # notice nnx.jit, not jax.jit
        def update_step(policy, optimizer, samples, Q_weights, prior_lps):
            (loss, (kld, ess)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(policy, samples, Q_weights, prior_lps)
            optimizer.update(grads)
            return policy, optimizer, loss, kld, ess
        
        self.policy.eval()
        initial_samples, _ = self.policy.sample(
            jax.random.split(self.rng, self.num_MC_samples)
        )
        samples_history = samples_history.at[0].set(initial_samples)
        # Q_history = Q_history.at[0].set(
        #     get_Q_weights(compute_value_functions_jit(samples_history[0]), samples_history[0])
        #     # get_Q_weights(A, samples_history[0])
        # )
        weights_history = weights_history.at[0].set(get_weights(samples_history[0]))
        prior_history = prior_history.at[0].set(prior_likelihood(samples_history[0]))
        values_history = values_history.at[0].set(self.fg(samples_history[0]))
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
        
        def epoch_step(carry, epoch):
            policy, optimizer, rng, samples_history, values_history, losses_history, Q_history, prior_history, kld_history, ess_history, weights_history = carry
            samples = samples_history[epoch]
            if self.replay_buffer:
                samples = samples_history.reshape(-1, self.n_variables)

            # Run inner optimization loop: training (grads)
            policy.train()
            init_carry = (policy, optimizer, samples, epoch)
            (policy, optimizer, _, _), (losses, klds, ess) = scan_inner(
                init_carry, jnp.arange(self.num_inner_iter)
            )
            
            # Sample from model for next epoch: internally sets eval (no grads)
            rng, sample_rng = jax.random.split(rng)
            policy.eval()
            samples, _ = policy.sample(
                jax.random.split(sample_rng, self.num_MC_samples)
            )
            
            # Update history arrays functionally
            samples_history = samples_history.at[epoch+1].set(samples)
            values_history = values_history.at[epoch+1].set(self.fg(samples))
            losses_history = losses_history.at[epoch+1].set(losses[-1])
            kld_history = kld_history.at[epoch+1].set(klds[-1])
            ess_history = ess_history.at[epoch+1].set(ess[-1])
            # Q_history = Q_history.at[epoch+1].set(
            #     # get_Q_weights(A, samples)
            #     get_Q_weights(compute_value_functions_jit(samples), samples)
            # )
            weights_history = weights_history.at[epoch+1].set(get_weights(samples))
            prior_history = prior_history.at[epoch+1].set(prior_likelihood(samples))
            
            return (policy, optimizer, rng, samples_history, values_history, losses_history, Q_history, prior_history, kld_history, ess_history, weights_history), None
        
        # Run epoch loop
        init_carry = (
            self.policy, nnx_optimizer, self.rng,
            samples_history, values_history, losses_history, Q_history, prior_history, kld_history, ess_history, weights_history
        )
        (self.policy, _, self.rng, samples_history, values_history, losses_history, Q_history, prior_history, kld_history, ess_history, weights_history), _ = nnx.scan(
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
        if self.use_prior != 'none':
            results['prior'] = prior_history
            if 'kld' in self.use_prior:
                results['kld'] = kld_history
            elif 'weight' in self.use_prior:
                # TODO: revise this based on weight scheme; maybe use per-node beta prior too...
                results['prior_weights'] = jnp.exp(prior_history / self.beta_prior)
                results['weights'] = results['prior_weights'] * weights_history
        
        return results
