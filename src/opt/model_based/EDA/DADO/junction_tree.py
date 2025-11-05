import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax as ox
from optax.contrib import muon
from typing import Callable, Tuple
import networkx as nx
from functools import partial
from time import time

from src.opt.model_based.EDA.DADO.base import DADO
from src.models.density.MLP.junction_tree import JunctionTreeDensityCliqueMLPs
from src.models.density.transformer.junction_tree import JunctionTreeDensityTransformer, JunctionTreeDensityCliqueTransformers
# from src.models.regression.MLP import IndependentRegressionMLP, JointRegressionMLP
from src.decomposition.fgm.tabular.base import TabularFunctionalJunctionTree
from src.decomposition.fgm.mlp.base import MLPFunctionalJunctionTree

from src.utils import count_params

WEIGHT_CLIP = 1e30 # clip weights to avoid overflow

class JunctionTreeDADO(DADO):
    """EDA using function decomposition and junction tree-structured density models."""
    
    def __init__(self, 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.fg, (TabularFunctionalJunctionTree, MLPFunctionalJunctionTree)), "JunctionTreeDADO requires a TabularFunctionalJunctionTree or MLPFunctionalJunctionTree."
    
    def precompute(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Train the density model using gradient-based optimization."""
        start_time = time()
        if self.use_transformer:
            self.policy = JunctionTreeDensityTransformer(
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
            self.policy = JunctionTreeDensityCliqueMLPs(
                self.fg,
                self.n_states,
                self.hidden_dims,
                rngs=self.rng
            )
            optimizer = ox.adamw
        if self.verbose: print(f"\t{self.policy.__class__.__name__} # params: {count_params(self.policy)}")
        self.fg.eval()
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
            prior_likelihood = lambda x: jnp.zeros((x.shape[0], len(self.fg.index_to_nodes)))

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
        ess_history = jnp.zeros((self.num_epochs + 1,))
        Q_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples, len(self.fg.index_to_nodes)))
        prior_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples, len(self.fg.index_to_nodes)))
        weights_history = jnp.zeros((self.num_epochs + 1, self.num_MC_samples, len(self.fg.index_to_nodes)))
        
        # Convert lists to tuples for JAX static arguments 
        children_tuple = tuple(tuple(child_list) for child_list in self.fg.children)
        parents_tuple = tuple(self.fg.parents)
        node_order_tuple = tuple(self.fg.node_order)
        index_to_nodes = tuple(tuple(sorted(node_set)) for node_set in self.fg.index_to_nodes)

        # jit these once model is done training; can't use decorator b/c will do @ init.
        if isinstance(self.fg, MLPFunctionalJunctionTree):
            self.fg.node_function_jax = jax.jit(self.fg.node_function)
            self.fg.edge_function_jax = jax.jit(self.fg.edge_function)

        # Helper function to encode clique state from assignment
        def encode_clique_state(clique_vars, assignment):
            """Encode assignment of variables in a clique to a single state index."""
            clique_states = jnp.zeros(assignment.shape[0], dtype=jnp.int32)
            for j, var in enumerate(clique_vars):
                radix = jnp.prod(jnp.array([self.n_states[clique_vars[k]] for k in range(j)])) if j > 0 else 1
                clique_states += assignment[:, var] * radix
            return clique_states
        
        # NOTE: nontrivial to make this a scan / fori_loop, skipping for now.
        def compute_value_functions(
                fg, samples, 
                V_samples, # [n_samples, n_cliques] - V values for each sample's clique assignment
                Q_samples, # [n_samples, n_cliques] - Q values for each sample's clique assignment
                node_order, parents, children, root, 
                index_to_nodes, 
            ):
            # NOTE: Each sample determines the specific clique states we need to compute
            # NOTE: we start w/ V = f(x_clique) for each clique, already initialized
            for clique in node_order[::-1]:  # Process each clique from leaves --> root
                # NOTE: receive messages from children cliques (sample-based)
                # V(x_clique) += \sum_{child_clique \in children(clique)} E_{j \sim p(child_clique|clique)} Q(clique, j)
                if len(children[clique]) > 0:
                    clique_vars = list(index_to_nodes[clique])
                    # NOTE: only used for vmapping + identifying unique parent states.
                    clique_states = encode_clique_state(clique_vars, samples)

                    def compute_update_for_parent_state(p_state, child_clique):
                        # Only use child samples where parent_clique == p_state
                        relevant_mask = (clique_states == p_state)
                        n_relevant = relevant_mask.sum()
                        relevant_q_vals = jnp.where(
                            relevant_mask,
                            Q_samples[:, child_clique], # Q(x_parent, x_child); i.e., Q(parent==p, child)
                            0.0
                        )
                        return jax.lax.select( # NOTE: if backprop, nan would be problematic
                            n_relevant > 0,
                            relevant_q_vals.sum() / n_relevant,
                            0.0
                        ) # mean over relevant samples
                    
                    # depending on the graph, may be better not to vmap this. 
                    V_samples = V_samples.at[:, clique].add(
                        jax.vmap(
                            lambda child_clique: jax.vmap(
                                lambda p_state: compute_update_for_parent_state(p_state, child_clique)
                            )(clique_states) # [n_samples]
                        )(jnp.array(children[clique]).astype(int)).sum(axis=0) # sum over children
                    )
                    # NOTE: if parent state is same in multiple samples they will have same V

                # NOTE: pass messages up to parent clique
                # Q(clique, x_parent_clique, x_clique) = f(x_parent_clique, x_clique) + V(x_clique)
                parent_clique = parents[clique]
                if parent_clique != -1:  # skip root clique
                    if isinstance(fg, TabularFunctionalJunctionTree):
                        # For tabular graphs, use precomputed edge function data
                        edge_key = (parent_clique, clique)
                        edge_data = edge_function_data[edge_key]
                        
                        def compute_edge_for_sample(s):
                            parent_vars = edge_data['parent_vars']
                            child_vars = edge_data['child_vars']
                            parent_state = s[parent_vars]
                            child_state = s[child_vars]
                            
                            # Compute joint state index for edge function lookup
                            parent_n_states = edge_data['parent_n_states']
                            child_n_states = edge_data['child_n_states']
                            
                            parent_radix = jnp.cumprod(jnp.concatenate([jnp.array([1]), parent_n_states[:-1]]))
                            parent_idx = jnp.sum(parent_state * parent_radix)
                            
                            child_radix = jnp.cumprod(jnp.concatenate([jnp.array([1]), child_n_states[:-1]]))
                            child_idx = jnp.sum(child_state * child_radix)
                            
                            return edge_data['table'][parent_idx, child_idx]
                        
                        edge_values = jax.vmap(compute_edge_for_sample)(samples)
                    else:
                        curr_edge_fn = lambda x: fg.edge_function_jax((parent_clique, clique), x)
                        edge_values = jax.vmap(curr_edge_fn)(samples)
                    
                    Q_samples = Q_samples.at[:, clique].set(
                        V_samples[:, clique] + edge_values
                    )

            # Set root clique Q = V (no parent contribution)
            Q_samples = Q_samples.at[:, root].set(V_samples[:, root])
            # Make advantage instead. How much better is pos than avg decision from parent?
            # NOTE: b/c there are multiple children, advantage is not so straightforward.
            # Really, want Q(clique, x_parent_clique, x_clique) - E_xclique [Q(clique, x_parent_clique, x_clique)]
            A = jnp.zeros_like(Q_samples)
            A = A.at[:, root].set(V_samples[:, root])
            for child in range(len(node_order)):
                parent = parents[child]
                if parent != -1:
                    parent_vars = list(index_to_nodes[parent])
                    # NOTE: only used for vmapping + identifying unique parent states.
                    parent_states = encode_clique_state(parent_vars, samples) # [n_samples]

                    def compute_expected_Q(p_state):
                        # Only use child samples where parent_clique == p_state
                        relevant_mask = (parent_states == p_state)
                        n_relevant = relevant_mask.sum()
                        relevant_q_vals = jnp.where(
                            relevant_mask,
                            Q_samples[:, child], # Q(x_parent, x_child); i.e., Q(parent==p, child)
                            0.0
                        )
                        return jax.lax.select( # NOTE: if backprop, nan would be problematic
                            n_relevant > 0,
                            relevant_q_vals.sum() / n_relevant,
                            0.0
                        ) # mean over relevant samples
                    
                    # depending on the graph, may be better not to vmap this. 
                    V_parent = jax.vmap(
                        lambda p_state: compute_expected_Q(p_state)
                    )(parent_states) # [n_samples]
                    A = A.at[:, child].set( # (n_samples,)
                        Q_samples[:, child] - V_parent # (n_samples,)
                    )
            return A
        
        @jax.jit
        def get_weights(x):
            Q = compute_value_functions_jit(x)
            return jnp.clip(jnp.exp(
                Q / self.temp_Q
            ), 0., WEIGHT_CLIP)

        # Precompute static data for tabular graphs outside JIT
        if isinstance(self.fg, TabularFunctionalJunctionTree):
            clique_vars_list = [jnp.array(list(clique)) for clique in index_to_nodes]
            clique_n_states_list = [jnp.array([self.fg.original_n_states_list[var] for var in clique]) 
                                  for clique in index_to_nodes]
            node_function_values = [self.fg.node_functions[i].value for i in range(len(index_to_nodes))]
            
            # Precompute edge function data
            edge_function_data = {}
            for edge_key, edge_fn_table in self.fg.edge_functions.items():
                parent_clique, child_clique = edge_key
                edge_function_data[edge_key] = {
                    'table': edge_fn_table,
                    'parent_vars': jnp.array(list(index_to_nodes[parent_clique])),
                    'child_vars': jnp.array(list(index_to_nodes[child_clique])),
                    'parent_n_states': jnp.array([self.fg.original_n_states_list[var] for var in index_to_nodes[parent_clique]]),
                    'child_n_states': jnp.array([self.fg.original_n_states_list[var] for var in index_to_nodes[child_clique]])
                }

        @jax.jit
        def initialize_v_samples(samples):
            """Initialize V values for each sample based on clique node functions."""
            n_cliques = len(index_to_nodes)
            if isinstance(self.fg, TabularFunctionalJunctionTree):
                # For tabular graphs, use precomputed static data
                def compute_v_for_sample(s):
                    v_values = []
                    for clique_idx in range(n_cliques):
                        clique_vars = clique_vars_list[clique_idx]
                        node_state = s[clique_vars]
                        clique_n_states = clique_n_states_list[clique_idx]
                        radix_multipliers = jnp.cumprod(jnp.concatenate([jnp.array([1]), clique_n_states[:-1]]))
                        joint_state_idx = jnp.sum(node_state * radix_multipliers)
                        v_values.append(node_function_values[clique_idx][joint_state_idx])
                    return jnp.array(v_values)
                return jax.vmap(compute_v_for_sample)(samples)
            else:
                # For MLP graphs, use the existing approach
                return jax.vmap(
                    lambda s: jax.vmap(
                        lambda clique_idx: self.fg.node_function_jax(clique_idx, s)
                    )(jnp.arange(n_cliques)) # [n_samples]
                )(samples) # [n_cliques, n_samples]

        @jax.jit
        def compute_value_functions_jit(samples):
            V_samples = initialize_v_samples(samples)
            Q_samples = jnp.zeros((samples.shape[0], len(index_to_nodes)))
            
            return compute_value_functions(
                fg=self.fg,
                samples=samples, 
                V_samples=V_samples, 
                Q_samples=Q_samples,
                node_order=node_order_tuple, 
                parents=parents_tuple, 
                children=children_tuple,
                root=self.fg.root,
                index_to_nodes=index_to_nodes,
            )

        # Use clique_weighted_likelihood for junction tree
        def loss_fn(policy, samples, Q_weights, prior_lps):
            clique_lps = policy.clique_likelihood(samples) # (N, L)
            kld = jax.lax.max((clique_lps.sum(axis=-1) - prior_lps.sum(axis=-1)).mean(), 0.) # TODO: not quite on policy, but close enough?
            ess = (Q_weights.sum())**2 / (Q_weights**2).sum()
            assert clique_lps.shape == Q_weights.shape
            wll = (clique_lps * Q_weights)
            if 'weight' in self.use_prior: # use p0 * exp(Q) instead of just exp(Q). use beta as temperature.
                prior_probs = jnp.exp(prior_lps * self.beta_prior) # nonneg weights
                assert prior_probs.shape == Q_weights.shape # (N, L)
                return -(jax.vmap(jnp.dot)(Q_weights * prior_probs, clique_lps)).mean(), (kld, ess)
            elif 'kld' in self.use_prior: # mode-seeking direction: penalty if policy has mass outside of prior's support.
                return -(wll.sum(axis=-1)).mean() + self.beta_prior * kld, (kld, ess)
            return -(wll.sum(axis=-1)).mean(), (kld, ess)
        
        @nnx.jit # notice nnx.jit, not jax.jit
        def update_step(policy, optimizer, samples, Q_weights, prior_lps):
            (loss, (kld, ess)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(policy, samples, Q_weights, prior_lps)
            optimizer.update(grads)
            return policy, optimizer, loss, kld, ess
        
        if self.verbose: print(f"Time to precompute: {time() - start_time} seconds")
        start_time = time()
        
        self.policy.eval()
        initial_samples, _ = self.policy.sample(
            jax.random.split(self.rng, self.num_MC_samples)
        )
        if self.verbose: print(f"Time to sample initial samples: {time() - start_time} seconds")
        start_time = time()
        samples_history = samples_history.at[0].set(initial_samples)
        Q_history = Q_history.at[0].set(
            compute_value_functions_jit(samples_history[0]) # note Q is Q_weights here.
        )
        weights_history = weights_history.at[0].set(get_weights(samples_history[0]))
        prior_history = prior_history.at[0].set(prior_likelihood(samples_history[0]))
        values_history = values_history.at[0].set(self.fg(samples_history[0]))
        initial_loss, (initial_kld, initial_ess) = loss_fn(self.policy, samples_history[0], Q_history[0], prior_history[0])
        losses_history = losses_history.at[0].set(initial_loss)
        kld_history = kld_history.at[0].set(initial_kld)
        ess_history = ess_history.at[0].set(initial_ess)
        if self.verbose: print(f"Time to compute initial value functions: {time() - start_time} seconds")
        start_time = time()

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

            samples_history = samples_history.at[epoch+1].set(samples)
            values_history = values_history.at[epoch+1].set(self.objective.query(samples))
            losses_history = losses_history.at[epoch+1].set(losses[-1])
            kld_history = kld_history.at[epoch+1].set(klds[-1])
            ess_history = ess_history.at[epoch+1].set(ess[-1])
            # Q_history = Q_history.at[epoch+1].set(
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
