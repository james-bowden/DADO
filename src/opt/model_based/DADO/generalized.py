import jax
import jax.numpy as jnp
from typing import Callable, Tuple, List, Dict, Any

from src.opt.model_based.EDA.DADO.base import DADO
from src.opt.model_based.EDA.DADO.independent import IndependentDADO
from src.opt.model_based.EDA.DADO.tree import TreeDADO
from src.opt.model_based.EDA.DADO.junction_tree import JunctionTreeDADO
from src.decomposition.graphs import DisconnectedGraph, Tree, JunctionTree, GeneralizedGraph
from src.decomposition.fgm.mlp.base import MLPFunctionalGraph, MLPGeneralizedFunctionalGraph
from src.decomposition.fgm.tabular.base import TabularFunctionalGraph, TabularGeneralizedFunctionalGraph
from src.problems.synthetic.graph.base import SyntheticGraphObjective

class GeneralizedDADO(DADO):
    """Generalized DADO that automatically chooses appropriate submodels based on graph structure."""
    # TODO: update to use all args from base class. also update to return dicts etc.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Verify that the objective has a GeneralizedFunctionalGraph
        if not isinstance(self.fg, GeneralizedGraph):
            raise ValueError("GeneralizedDADO requires an objective with a GeneralizedFunctionalGraph")

        # Use the existing graph structure from the GeneralizedFunctionalGraph
        self.submodels = []
        self.node_to_submodel = self.fg.node_to_subgraph  # Use existing mapping
        # self.submodel_node_indices = self.fg.subgraph_node_indices  # Use existing indices

        self._create_submodels()

    def _create_submodels(self):
        """Create appropriate submodels for each component using the existing functional subgraphs."""
        if isinstance(self.fg, MLPGeneralizedFunctionalGraph):
            functional_subgraphs = self.fg.model.functional_subgraphs
        elif isinstance(self.fg, TabularGeneralizedFunctionalGraph):
            functional_subgraphs = self.fg.functional_subgraphs
        else:
            raise ValueError(f"Unknown generalized functional graph type: {type(self.fg)}")

        for subgraph_idx, (subgraph, functional_subgraph) in enumerate(zip(self.fg.subgraphs, functional_subgraphs)):
            # Create a mock objective that wraps the functional subgraph
            component_obj = self._create_component_objective(subgraph_idx, functional_subgraph)

            # Determine the appropriate submodel type based on the subgraph
            if isinstance(subgraph, DisconnectedGraph):
                # Use IndependentDADO for disconnected components
                submodel = IndependentDADO(
                    # n_variables=component_obj.n_variables,
                    n_states=component_obj.D,
                    num_layers=self.num_layers,
                    dim_ff=self.dim_ff,
                    num_heads=self.num_heads,
                    dim_attn=self.dim_attn,
                    use_transformer=self.use_transformer,
                    num_epochs=self.num_epochs,
                    num_inner_iter=self.num_inner_iter,
                    num_MC_samples=self.num_MC_samples,
                    learning_rate=self.learning_rate,
                    objective_fn=component_obj,
                    seed=self.seed + subgraph_idx if self.seed is not None else None,
                    verbose=self.verbose,
                    use_prior=self.use_prior,
                    beta_prior=self.beta_prior,
                    prior_data=self.prior_data,
                    replay_buffer=self.replay_buffer,
                    temp_Q=self.temp_Q,
                )
            elif isinstance(subgraph, JunctionTree) or isinstance(subgraph, Tree):
                # Use JunctionTreeDADO for general graphs
                submodel = JunctionTreeDADO(
                    # n_variables=component_obj.n_variables,
                    n_states=component_obj.D,
                    # n_states=self.functional_subgraph.n_states,
                    num_layers=self.num_layers,
                    dim_ff=self.dim_ff,
                    num_heads=self.num_heads,
                    dim_attn=self.dim_attn,
                    use_transformer=self.use_transformer,
                    num_epochs=self.num_epochs,
                    num_inner_iter=self.num_inner_iter,
                    num_MC_samples=self.num_MC_samples,
                    learning_rate=self.learning_rate,
                    objective_fn=component_obj,
                    seed=self.seed + subgraph_idx if self.seed is not None else None,
                    verbose=self.verbose,
                    use_prior=self.use_prior,
                    beta_prior=self.beta_prior,
                    prior_data=self.prior_data,
                    replay_buffer=self.replay_buffer,
                    temp_Q=self.temp_Q,
                )
            elif isinstance(subgraph, Tree):
                # Use TreeDADO for tree components
                submodel = TreeDADO(
                    # n_variables=component_obj.n_variables,
                    n_states=component_obj.D,
                    num_layers=self.num_layers,
                    dim_ff=self.dim_ff,
                    num_heads=self.num_heads,
                    dim_attn=self.dim_attn,
                    use_transformer=self.use_transformer,
                    num_epochs=self.num_epochs,
                    num_inner_iter=self.num_inner_iter,
                    num_MC_samples=self.num_MC_samples,
                    learning_rate=self.learning_rate,
                    objective_fn=component_obj,
                    seed=self.seed + subgraph_idx if self.seed is not None else None,
                    verbose=self.verbose,
                    use_prior=self.use_prior,
                    beta_prior=self.beta_prior,
                    prior_data=self.prior_data,
                    replay_buffer=self.replay_buffer,
                    temp_Q=self.temp_Q,
                )
            else:
                raise ValueError(f"Unknown subgraph type: {type(subgraph)}")

            self.submodels.append(submodel)

    def _create_component_objective(self, subgraph_idx: int, functional_subgraph):
        """Initialize a synthetic objective for a functional subgraph."""         
        obj = SyntheticGraphObjective(
            D=functional_subgraph.n_states_list,
            fg=functional_subgraph,
            weight_fn_name='id',
        )
        obj.subgraph_idx = subgraph_idx
        obj.obj_name = f"{self.objective.obj_name}_sub{subgraph_idx}"
        return obj

    def stitch(self, submodel_samples: List[jnp.ndarray]) -> jnp.ndarray:
        """Combine samples from submodels back into the original node order, for arbitrary leading dimensions.
        Each submodel's output is placed into the correct indices in the last dimension, as specified by self.submodel_node_indices.
        """
        leading_shape = submodel_samples[0].shape[:-1] # must be shared by all submodel samples
        full_samples = jnp.zeros(leading_shape + (len(self.n_states),), dtype=jnp.int32)

        for submodel_idx, samples in enumerate(submodel_samples):
            node_indices = self.fg.subgraph_node_indices[submodel_idx]
            full_samples = full_samples.at[..., node_indices].set(samples)
        return full_samples

    def precompute(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Train all submodels and combine their histories."""
        self.results_dicts = []
        self.results = {}
        shared_keys = set()

        for submodel in self.submodels:
            results_dict = submodel.precompute()
            self.results_dicts.append(results_dict)
            if len(shared_keys) == 0:
                shared_keys = set(results_dict.keys())
            else:
                shared_keys = shared_keys.intersection(set(results_dict.keys()))
        shared_keys.remove('samples')
        self.results['samples'] = self.stitch([d['samples'] for d in self.results_dicts])
        shared_keys.remove('values')
        self.results['values'] = jnp.array([d['values'] for d in self.results_dicts]).sum(axis=0) + self.objective.fg.model.constant.value
        shared_keys.remove('losses')
        self.results['losses'] = jnp.array([d['losses'] for d in self.results_dicts]).sum(axis=0)
        if 'kld' in shared_keys:
            shared_keys.remove('kld')
            self.results['kld'] = jnp.array([d['kld'] for d in self.results_dicts]).sum(axis=0)
        shared_keys.remove('Q')
        self.results['Q'] = jnp.concatenate([d['Q'] for d in self.results_dicts], axis=-1) # could stitch... but downstream not using order.
        if 'ess' in shared_keys:
            shared_keys.remove('ess')
            self.results['ess'] = jnp.concatenate([d['ess'] for d in self.results_dicts], axis=-1)
        if 'prior' in shared_keys:
            shared_keys.remove('prior')
            self.results['prior'] = jnp.concatenate([d['prior'] for d in self.results_dicts], axis=-1)
        if 'prior_weights' in shared_keys:
            shared_keys.remove('prior_weights')
            self.results['prior_weights'] = jnp.concatenate([d['prior_weights'] for d in self.results_dicts], axis=-1)
        if 'weights' in shared_keys:
            shared_keys.remove('weights')
            self.results['weights'] = jnp.concatenate([d['weights'] for d in self.results_dicts], axis=-1)
        for key in shared_keys:
            self.results[key] = jnp.array([results_dict[key] for results_dict in self.results_dicts])
        
        # # Combine histories by stitching samples and summing values/losses
        # combined_samples_history = self.stitch(self.all_samples_histories)
        # combined_values_history = jnp.array(self.all_values_histories).sum(axis=0) + self.objective.fg.model.constant.value
        # combined_losses_history = jnp.array(self.all_losses_histories).sum(axis=0)
        # combined_entropy_history = jnp.array(self.all_entropy_histories).sum(axis=0)
        # combined_Q_history = jnp.concatenate(self.all_Q_histories, axis=-1)

        if self.submodels:
            self.policy = self.submodels # NOTE: may have weird behavior if used externally. using for saving.
            self.model = self.policy

        # return combined_samples_history, combined_values_history, combined_losses_history, combined_entropy_history, combined_Q_history
        return self.results

    def get_solutions(self, n_points: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample from all submodels and combine the solutions."""
        submodel_samples, submodel_values = [], []

        for submodel in self.submodels:
            samples, values = submodel.get_solutions(n_points)
            submodel_samples.append(samples)
            submodel_values.append(values)

        # Stitch samples together
        combined_samples = self.stitch(submodel_samples)
        combined_values = jnp.array(submodel_values).sum(axis=0) + self.objective.fg.model.constant.value

        # Evaluate objective on combined samples
        objective_values = self.objective.query(combined_samples)
        if not (objective_values == combined_values).all():
            print(f"[Warning] GeneralizedGraph Objective values {objective_values} do not match sum of values of stitched samples {combined_values}")
            print(f"Constant: {self.objective.fg.model.constant.value}")

        return combined_samples, objective_values

    def get_mode(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get the mode from all submodels and combine them."""
        submodel_modes, value = [], self.objective.fg.model.constant.value

        for submodel in self.submodels:
            mode, v = submodel.get_mode()
            submodel_modes.append(mode[None, :])  # Add batch dimension
            value += v

        combined_mode = self.stitch(submodel_modes).squeeze()  # Remove batch dimension
        objective_value = self.objective.query(combined_mode[None, :]).squeeze()
        if not (objective_value == value):
            print(f"[Warning] GeneralizedGraph Objective values {objective_value} do not match sum of values of stitched samples {value}")
            print(f"Constant: {self.objective.fg.model.constant.value}")
        
        return combined_mode, objective_value
