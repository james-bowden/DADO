import numpy as np
import jax
import jax.numpy as jnp
from typing import Union, List, Tuple, Optional, Dict, Callable
from functools import partial

import optax as ox
from flax import nnx

from src.decomposition.graphs import Graph, Tree, JunctionTree, GeneralizedGraph, DisconnectedGraph
from src.models.regression.MLP import GraphRegressionMLP, GraphRegressionMLPEmbedded

from src.utils import fix_nbatch_smallest_batch_size


class MLPFunctionalGraph(Graph, nnx.Module):
    """A graph where each node/edge is represented by a single GraphRegressionMLP."""
    
    def __init__(self, 
                 n_nodes: int, 
                 edges: Union[List[Tuple[int, int]], np.ndarray],
                 n_states_list: List[int],
                 hidden_dims: List[int],
                 *,
                 batch_size: int = -1,
                 batch_size_graph: int = -1,
                 embedding_dim: int = 0,
                 rngs: jax.random.PRNGKey,
                 **kwargs):
        """
        Initialize a functional graph with n_nodes nodes and a single GraphRegressionMLP.
        
        Args:
            n_nodes: Number of nodes in the graph
            edges: Optional edge list or adjacency matrix
            n_states_list: Number of possible states for each variable
            hidden_dims: List of hidden layer dimensions for the MLP
            rngs: Random number generators for initialization
        """
        super().__init__(n_nodes, edges, **kwargs)
        self.n_states_list = n_states_list
        assert len(self.n_states_list) == n_nodes, f"Number of n_states_list {len(self.n_states_list)} does not match number of nodes {n_nodes}"
        
        self.batch_size = batch_size
        self.batch_size_graph = batch_size_graph
        if embedding_dim <= 0:
            self.model = GraphRegressionMLP(n_states_list, hidden_dims, rngs=rngs)#, scaled=False, nonneg=True)
        else:
            self.model = GraphRegressionMLPEmbedded(n_states_list, hidden_dims, embedding_dim, rngs=rngs)#, scaled=False, nonneg=True)
        self.global_nl = lambda x: x
        
        # Pre-allocate index arrays to avoid concatenation during forward pass
        node_inds = self._create_all_node_inputs()
        edge_inds = self._create_all_edge_inputs()
        all_inds = jnp.concatenate([node_inds, edge_inds], axis=0)
        self.all_inds = nnx.Variable(all_inds)
        
        # Pre-batch indices for efficient scanning when batch_size_graph > 0
        if batch_size_graph > self.all_inds.shape[0]: # skip batching if batch_size_graph is larger than the number of indices
            batch_size_graph = -1
        if batch_size_graph > 0:
            n_indices = all_inds.shape[0]
            batch_size_graph, n_batches = fix_nbatch_smallest_batch_size(batch_size_graph, n_indices)
            total_padded = n_batches * batch_size_graph
            
            if total_padded > n_indices:
                # Pad indices to make evenly divisible batches with zeros (inactive nodes/edges)
                padded_inds = jnp.pad(all_inds, ((0, total_padded - n_indices), (0, 0)), mode='constant', constant_values=False)
            else:
                padded_inds = all_inds
                
            self.all_inds_batched = nnx.Variable(padded_inds.reshape(n_batches, batch_size_graph, -1))
            self.n_graph_batches = n_batches
            self.original_n_indices = n_indices
        else:
            self.all_inds_batched = None
            self.n_graph_batches = 0
            self.original_n_indices = 0

    @classmethod
    def random(cls, n_nodes: int, n_edges: int, n_states_list: List[int], hidden_dims: List[int], *, batch_size: int = -1, batch_size_graph: int = -1, embedding_dim: int = 0, rngs: jax.random.PRNGKey) -> 'MLPFunctionalGraph':
        """
        Create a random functional graph with n_nodes nodes and n_edges edges.
        
        Args:
            n_nodes: Number of nodes
            n_edges: Number of edges to sample
            n_states_list: Number of possible states for each node
            hidden_dims: List of hidden layer dimensions for the MLP
            rngs: Random number generators
        """
        graph = super().random(n_nodes, n_edges)
        return cls(n_nodes, graph.get_edge_list(), n_states_list, hidden_dims, batch_size=batch_size, batch_size_graph=batch_size_graph, embedding_dim=embedding_dim, rngs=rngs)

    @classmethod
    def from_graph(cls, graph: Graph, n_states_list: List[int], hidden_dims: List[int], *, batch_size: int = -1, batch_size_graph: int = -1, embedding_dim: int = 0, rngs: jax.random.PRNGKey) -> 'MLPFunctionalGraph':
        """
        Create a functional graph from a graph.
        """
        return cls(graph.n_nodes, graph.get_edge_list(), n_states_list, hidden_dims, batch_size=batch_size, batch_size_graph=batch_size_graph, embedding_dim=embedding_dim, rngs=rngs)

    def _create_node_input(self, node: int) -> jnp.ndarray:
        """Create input for node function evaluation."""
        inds = jnp.zeros(len(self.n_states_list)).astype(bool)
        inds = inds.at[node].set(True)  # Only this node is active
        return inds

    def _create_edge_input(self, edge: Tuple[int, int]) -> jnp.ndarray:
        """Create input for edge function evaluation."""
        inds = jnp.zeros(len(self.n_states_list)).astype(bool)
        inds = inds.at[edge[0]].set(True)  # Both nodes in edge are active
        inds = inds.at[edge[1]].set(True)
        return inds

    def _create_all_node_inputs(self) -> jnp.ndarray:
        """Create batched inputs for all node function evaluations."""
        node_inds = jax.vmap(self._create_node_input)(jnp.arange(self.n_nodes))
        return node_inds
    
    def _create_all_edge_inputs(self) -> jnp.ndarray:
        """Create batched inputs for all edge function evaluations."""
        if len(self.edges) == 0:
            return jnp.empty((0, len(self.n_states_list)))
        
        edges_array = jnp.array(self.edges)
        edge_inds = jax.vmap(self._create_edge_input)(edges_array)
        return edge_inds
    
    @nnx.jit # NOTE: was below vmap, trying above but unsure. I think before it would vectorize compilation...(bad)
    @partial(nnx.vmap, in_axes=(None, 0))
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the function for a single input."""
        # Use pre-allocated index arrays
        if self.batch_size_graph <= 0:
            # Single batched call to the model
            all_inds = self.all_inds.value
            out = self.model(all_inds, x).sum()
        else:
            print(f"graph batch size: {self.batch_size_graph}")
            # Use pre-batched indices with simple loop for memory efficiency
            out = jnp.zeros((self.n_graph_batches*self.batch_size_graph,))
            for i, batch_inds in enumerate(self.all_inds_batched.value):
                out = out.at[i*self.batch_size_graph:(i+1)*self.batch_size_graph].set(
                    self.model(batch_inds, x)
                )
            out = out[:self.original_n_indices].sum()
        # NOTE: using sigmoid here isn't necessary, but works nicely b/c we always scale our training data
        # to a similar range. It may be undesirable for optimization purposes, but because it's monotonic,
        # it's theoretically fine.
        return nnx.sigmoid(out)
    
    def evaluate_single(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the function for a single input."""
        if self.batch_size_graph <= 0:
            call_model = lambda x_single: nnx.sigmoid(self.model(self.all_inds.value, x_single).sum())
        else:
            def call_model(x_single):
                out = jnp.zeros((self.n_graph_batches*self.batch_size_graph,))
                for i, batch_inds in enumerate(self.all_inds_batched.value):
                    out = out.at[i*self.batch_size_graph:(i+1)*self.batch_size_graph].set(
                        self.model(batch_inds, x_single)
                    )
                return nnx.sigmoid(out[:self.original_n_indices].sum())
        return call_model(x)
    
    def evaluate_seq(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the function sequentially."""
        results = jnp.zeros((x.shape[0],))
        for i in range(x.shape[0]):
            results = results.at[i].set(self.evaluate_single(x[i]))
        return results
        
    @staticmethod # NOTE: important!! avoid tracing / mutation.
    def _split_merge(fg: 'MLPFunctionalGraph') -> jnp.ndarray:
        graphdef, state = nnx.split(fg)
        new_fg = nnx.merge(graphdef, state)
        return new_fg
    
    def evaluate_batched(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.batch_size <= 0 or self.batch_size >= x.shape[0]:
            return self(x)
        else:
            n_samples = x.shape[0]
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            total_padded = n_batches * self.batch_size
            results = jnp.zeros((total_padded,))
            
            if total_padded > n_samples: # zero-pad
                x_padded = jnp.pad(x, ((0, total_padded - n_samples), (0, 0)), mode='constant')
            else: x_padded = x
            x_batched = x_padded.reshape(n_batches, self.batch_size, -1)
            
            for i, x_batch in enumerate(x_batched): # simple loop instead of scan
                results = results.at[i*self.batch_size:(i+1)*self.batch_size].set(
                    self(x_batch)
                )
            return results[:n_samples] # unpad

    def node_function(self, node: int, x: jnp.ndarray) -> jnp.ndarray:
        return self._node_function_jax(node, x)
    
    def edge_function(self, edge: Tuple[int, int], x: jnp.ndarray) -> jnp.ndarray:
        return self._edge_function_jax(edge, x)
    
    def _node_function_jax(self, node: int, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the node function for a single input."""
        inds = jnp.zeros((1, len(self.n_states_list))).astype(bool) # call vmapped model w/ batch size=1
        inds = inds.at[0, node].set(True)  # Only this node is active
        return self.model(inds, x).squeeze()
    
    def _edge_function_jax(self, edge: Tuple[int, int], x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the edge function for a single input."""
        inds = jnp.zeros((1, len(self.n_states_list))).astype(bool) # call vmapped model w/ batch size=1
        inds = inds.at[0, edge[0]].set(True)  # Both nodes in edge are active
        inds = inds.at[0, edge[1]].set(True)
        return self.model(inds, x).squeeze()
    
    @staticmethod
    def fit(
        fg: 'MLPFunctionalGraph', 
        x: jnp.ndarray, y: jnp.ndarray,
        n_iter_inner: int = 20,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        seed: int = 48,
    ):
        """
        Fit the MLP functional graph to the given values.
        """
        optimizer = ox.adamw(learning_rate)
        nnx_optimizer = nnx.Optimizer(fg.model, optimizer)
        key = jax.random.PRNGKey(seed)

        @nnx.jit
        def train_step(model, optimizer, x_batch, y_batch):
            def loss_fn(model):
                y_pred = model(x_batch.astype(int))
                return ((y_pred - y_batch) ** 2).mean()

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(grads)

            return model, optimizer, loss
        
        def inner_step(carry, _):
            model, nnx_opt, x_batch, y_batch = carry
            model, nnx_opt, loss = train_step(model, nnx_opt, x_batch, y_batch)
            return (model, nnx_opt, x_batch, y_batch), loss
        
        scan_inner = nnx.scan(
            f=inner_step, length=n_iter_inner
        )

        def epoch_step(carry, _):
            model, nnx_opt, key = carry
            
            # Subsample without replacement
            key, subkey = jax.random.split(key)
            n_samples = x.shape[0]
            batch_size = fg.batch_size if fg.batch_size > 0 else n_samples
            batch_size = min(batch_size, n_samples)  # Don't exceed available samples
            
            indices = jax.random.choice(subkey, n_samples, shape=(batch_size,), replace=False)
            x_batch = x[indices]
            y_batch = y[indices]
            
            (model, nnx_opt, _, _), losses = scan_inner(
                (model, nnx_opt, x_batch, y_batch), 
                jnp.arange(n_iter_inner)
            )
            return (model, nnx_opt, key), losses[-1]

        fg.train()
        # NOTE: use fg instead of fg.model, b/c we want to use fg.__call__ method.
        (fg, nnx_optimizer, _), losses = nnx.scan(f=epoch_step, length=n_epochs)(
            (fg, nnx_optimizer, key), jnp.arange(n_epochs)
        )
        fg.eval()
        
        return losses


class MLPFunctionalTree(MLPFunctionalGraph, Tree, nnx.Module):
    """A tree where each node/edge is represented by a single GraphRegressionMLP."""

    def __init__(self, n_nodes: int, edges: Union[List[Tuple[int, int]], np.ndarray],
                 n_states_list: List[int],
                 hidden_dims: List[int],
                 *,
                 batch_size: int = -1,
                 batch_size_graph: int = -1,
                 embedding_dim: int = 0,
                 rngs: jax.random.PRNGKey,
                 root: Optional[int] = None,
                 **kwargs):
        """
        Initialize a functional tree with n_nodes nodes and a single GraphRegressionMLP.
        
        Args:
            n_nodes: Number of nodes in the tree
            edges: Optional edge list or adjacency matrix
            n_states_list: Number of possible states for each node
            hidden_dims: List of hidden layer dimensions for the MLP
            rngs: Random number generators for initialization
            root: Optional root node index
        """
        # Initialize both parent classes
        Tree.__init__(self, n_nodes, edges, root, **kwargs)
        MLPFunctionalGraph.__init__(self, n_nodes, edges, n_states_list, hidden_dims, batch_size=batch_size, batch_size_graph=batch_size_graph, embedding_dim=embedding_dim, rngs=rngs)
    
    @classmethod
    def random(cls, n_nodes: int, n_states_list: List[int], hidden_dims: List[int], *, 
               batch_size: int = -1, batch_size_graph: int = -1,
               embedding_dim: int = 0,
               rngs: jax.random.PRNGKey, root: Optional[int] = None) -> 'MLPFunctionalTree':
        """
        Create a random functional tree with n_nodes nodes.
        
        Args:
            n_nodes: Number of nodes
            n_states_list: Number of possible states for each node
            hidden_dims: List of hidden layer dimensions for the MLP
            rngs: Random number generators
            root: Optional root node index
        """
        tree = Tree.random(n_nodes, root=root)
        return cls(n_nodes, tree.get_edge_list(), n_states_list, hidden_dims, batch_size=batch_size, batch_size_graph=batch_size_graph, embedding_dim=embedding_dim, rngs=rngs, root=tree.root)


class MLPFunctionalJunctionTree(MLPFunctionalGraph, JunctionTree, nnx.Module):
    """A junction tree where each clique is represented by a single GraphRegressionMLP over original variables."""
    
    def __init__(self, n_nodes: int, edges: Union[List[Tuple[int, int]], np.ndarray],
                 index_to_nodes: List[set], nodes_to_index: Dict[set, int],
                 n_states_list: List[int],
                 hidden_dims: List[int],
                 *,
                 batch_size: int = -1,
                 batch_size_graph: int = -1,
                 embedding_dim: int = 0,
                 rngs: jax.random.PRNGKey,
                 root: Optional[int] = None,
                 **kwargs):
        """
        Initialize a functional junction tree with a single GraphRegressionMLP over original variables.
        
        Args:
            n_nodes: Number of nodes in the junction tree
            edges: Edge list or adjacency matrix on cliques, not original nodes.
            index_to_nodes: List of sets representing original graph nodes in each JT node
            nodes_to_index: Dictionary mapping node sets to indices
            n_states_list: Number of possible states for each original variable
            hidden_dims: List of hidden layer dimensions for the MLP
            rngs: Random number generators for initialization
            root: Optional root node index
        """
        JunctionTree.__init__(self, n_nodes, edges, index_to_nodes, nodes_to_index, root)
        self.original_n_states_list = n_states_list
        # Initialize MLP over original variables, not JT nodes
        self.n_states_list = n_states_list
        self.batch_size = batch_size
        self.batch_size_graph = batch_size_graph
        if embedding_dim <= 0:
            self.model = GraphRegressionMLP(n_states_list, hidden_dims, rngs=rngs)#, scaled=False, nonneg=True)
        else:
            self.model = GraphRegressionMLPEmbedded(n_states_list, hidden_dims, embedding_dim, rngs=rngs)#, scaled=False, nonneg=True)
        self.global_nl = lambda x: x
        
        # Convert index_to_nodes to JAX-compatible format
        max_original_nodes = len(n_states_list)
        self.clique_masks = nnx.Variable(jnp.zeros((n_nodes, max_original_nodes), dtype=bool))
        for i, node_set in enumerate(index_to_nodes):
            mask = jnp.zeros(max_original_nodes, dtype=bool)
            if node_set:  # Check if set is not empty
                node_indices = jnp.array(list(node_set))
                mask = mask.at[node_indices].set(True)
            self.clique_masks.value = self.clique_masks.value.at[i].set(mask)
        
        # Precompute edges array for JAX-compatible edge lookup
        # NOTE: assuming has edges -- generalized graph takes care of this.
        self.edges_array = nnx.Variable(jnp.array(self.edges))  # Shape: (n_edges, 2)
        
        # Pre-allocate index arrays to avoid concatenation during forward pass
        edge_inds = self._create_all_edge_inputs()
        all_inds = jnp.concatenate([self.clique_masks.value, edge_inds], axis=0)
        self.all_inds = nnx.Variable(all_inds)
        
        # Pre-batch indices for efficient scanning when batch_size_graph > 0
        if batch_size_graph > self.all_inds.shape[0]: # skip batching if batch_size_graph is larger than the number of indices
            batch_size_graph = -1
        if batch_size_graph > 0:
            n_indices = all_inds.shape[0]
            batch_size_graph, n_batches = fix_nbatch_smallest_batch_size(batch_size_graph, n_indices)
            total_padded = n_batches * batch_size_graph
            
            if total_padded > n_indices:
                # Pad indices to make evenly divisible batches with zeros (inactive nodes/edges)
                padded_inds = jnp.pad(all_inds, ((0, total_padded - n_indices), (0, 0)), mode='constant', constant_values=False)
            else:
                padded_inds = all_inds
                
            self.all_inds_batched = nnx.Variable(padded_inds.reshape(n_batches, batch_size_graph, -1))
            self.n_graph_batches = n_batches
            self.original_n_indices = n_indices
        else:
            self.all_inds_batched = None
            self.n_graph_batches = 0
            self.original_n_indices = 0

    @classmethod
    def from_graph(cls, jt: JunctionTree, n_states_list: List[int], hidden_dims: List[int], *, batch_size: int = -1, batch_size_graph: int = -1, embedding_dim: int = 0, rngs: jax.random.PRNGKey) -> 'MLPFunctionalJunctionTree':
        """
        Create a functional junction tree from a junction tree.
        """
        return cls(jt.n_nodes, jt.get_edge_list(), jt.index_to_nodes, jt.nodes_to_index, n_states_list, hidden_dims, batch_size=batch_size, batch_size_graph=batch_size_graph, embedding_dim=embedding_dim, rngs=rngs)

    def _create_edge_input(self, edge: Tuple[int, int]) -> jnp.ndarray:
        """Create input for edge function evaluation using bitvector."""
        i_mask = self.clique_masks.value[edge[0]]
        j_mask = self.clique_masks.value[edge[1]]
        union_mask = jnp.logical_or(i_mask, j_mask)
        return union_mask.astype(bool) #, i_mask, j_mask

    def _create_all_edge_inputs(self) -> jnp.ndarray:
        """Create batched inputs for all edge function evaluations."""
        edge_inds = jax.vmap(
            lambda e: self._create_edge_input(e) # discard i_mask, j_mask
        )(self.edges_array)
        return edge_inds
    
    def _node_function_jax(self, node: int, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the node function for a single input."""
        clique_mask = self.all_inds.value[node] # precomputed
        return self.model(clique_mask[None, :], x).squeeze()
    
    def _edge_function_jax(self, edge: Tuple[int, int], x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the edge function for a single input."""
        edge_array = jnp.array([edge[0], edge[1]])
        # print(f"edge_array: {edge_array}, self.edges_array.value: {self.edges_array.value}")
        matches = jnp.all(self.edges_array.value == edge_array, axis=1)
        edge_idx = jnp.where(matches, size=1)[0][0]  # Get first (and should be only) match
        # print(f"edge_idx: {edge_idx}, matches: {matches.sum()}")
        edge_mask = self.all_inds.value[self.n_nodes + edge_idx] # precomputed
        return self.model(edge_mask[None, :], x).squeeze()


class MLPGeneralizedFunctionalGraph(GeneralizedGraph, nnx.Module):
    """A functional graph that automatically decomposes itself into MLP-based subcomponents."""
    
    def __init__(self, 
                 generalized_graph: GeneralizedGraph,
                 n_states_list: List[int],
                 hidden_dims: List[int],
                 *,
                 batch_size: int = -1,
                 batch_size_graph: int = -1,
                 embedding_dim: int = 0,
                 wt: Optional[jnp.ndarray] = None,
                 rngs: jax.random.PRNGKey,
                 **kwargs):
        """Initialize a generalized functional graph from a generalized graph and functional subgraphs."""
        # Copy structure from generalized graph
        super().__init__(generalized_graph.n_nodes, generalized_graph.get_edge_list(), verbose=False)
        self.n_states_list = n_states_list
        self.hidden_dims = hidden_dims
        self.subgraphs = generalized_graph.subgraphs
        self.node_to_subgraph = generalized_graph.node_to_subgraph
        self.subgraph_node_indices = generalized_graph.subgraph_node_indices
        self.batch_size = batch_size
        self.wt = wt
        if self.wt is not None: 
            self.wt = self.wt[None, :].astype(int)

        # Yield to subgraph nonlinearities.
        self.global_nl = lambda x: x
        
        # Init functional subgraphs
        functional_subgraphs = []
        for subgraph_idx, subgraph in enumerate(self.subgraphs):
            global_node_indices = self.subgraph_node_indices[subgraph_idx]
            sub_n_states_list = [self.n_states_list[i] for i in global_node_indices]
            
            # Create the appropriate functional subgraph
            if isinstance(subgraph, DisconnectedGraph):
                functional_subgraph = MLPFunctionalGraph.from_graph(
                    subgraph, sub_n_states_list, hidden_dims, batch_size=batch_size, batch_size_graph=batch_size_graph, embedding_dim=embedding_dim, rngs=rngs,
                )
            elif isinstance(subgraph, JunctionTree):
                functional_subgraph = MLPFunctionalJunctionTree.from_graph(
                    subgraph, sub_n_states_list, hidden_dims, batch_size=batch_size, batch_size_graph=batch_size_graph, embedding_dim=embedding_dim, rngs=rngs,
                )
            elif isinstance(subgraph, Tree):
                functional_subgraph = MLPFunctionalTree.from_graph(
                    subgraph, sub_n_states_list, hidden_dims, batch_size=batch_size, batch_size_graph=batch_size_graph, embedding_dim=embedding_dim, rngs=rngs, # no root
                )
            else:
                raise ValueError(f"Subgraph was neither DisconnectedGraph, Tree, nor JunctionTree. "
                               f"Subgraph type: {type(subgraph)}, edges: {subgraph.get_edge_list()}")
            
            functional_subgraphs.append(functional_subgraph)

        self.model = AggregateModel(
            functional_subgraphs, 
            self.global_nl, 
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the function by combining evaluations from all subgraphs."""
        r = self.model(x)
        if self.wt is not None: r -= self.model(self.wt).squeeze()
        return r

    def evaluate_batched(self, x: jnp.ndarray) -> jnp.ndarray:
        r = self.model.evaluate_batched(x)
        if self.wt is not None: r -= self.model.evaluate_batched(self.wt).squeeze()
        return r
    
    def evaluate_seq(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the function sequentially."""
        r = self.model.evaluate_seq(x)
        if self.wt is not None: r -= self.model.evaluate_seq(self.wt).squeeze()
        return r
    
    @staticmethod
    def fit( # NOTE: same as MLPFunctionalGraph.fit. Redefine b/c doesn't inherit. TODO: Should it?
        fg: 'MLPGeneralizedFunctionalGraph', 
        x: jnp.ndarray, y: jnp.ndarray,
        n_iter_inner: int = 20,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        seed: int = 48,
    ):
        """
        Fit the MLP functional graph to the given values.
        """
        optimizer = ox.adamw(learning_rate)
        nnx_optimizer = nnx.Optimizer(fg.model, optimizer)
        key = jax.random.PRNGKey(seed)

        @nnx.jit
        def train_step(model, optimizer, x_batch, y_batch):
            def loss_fn(model):
                y_pred = model(x_batch.astype(int))
                if fg.wt is not None: y_pred -= model(fg.wt).squeeze()
                return ((y_pred - y_batch) ** 2).mean()

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(grads)

            return model, optimizer, loss
        
        def inner_step(carry, _):
            model, nnx_opt, x_batch, y_batch = carry
            model, nnx_opt, loss = train_step(model, nnx_opt, x_batch, y_batch)
            return (model, nnx_opt, x_batch, y_batch), loss
        
        scan_inner = nnx.scan(
            f=inner_step, length=n_iter_inner
        )

        def epoch_step(carry, _):
            model, nnx_opt, key = carry
            
            # Subsample without replacement
            key, subkey = jax.random.split(key)
            n_samples = x.shape[0]
            batch_size = fg.batch_size if (fg.batch_size > 0 and fg.batch_size < n_samples) else n_samples
            
            indices = jax.random.choice(subkey, n_samples, shape=(batch_size,), replace=False)
            x_batch = x[indices]
            y_batch = y[indices]
            
            (model, nnx_opt, _, _), losses = scan_inner(
                (model, nnx_opt, x_batch, y_batch), 
                jnp.arange(n_iter_inner)
            )
            return (model, nnx_opt, key), losses[-1]

        fg.train()
        # NOTE: use fg.model b/c we want to use AggregateModel.__call__ method instead of fg.__call__.
        (fg.model, nnx_optimizer, _), losses = nnx.scan(f=epoch_step, length=n_epochs)(
            (fg.model, nnx_optimizer, key), jnp.arange(n_epochs)
        )
        fg.eval()
        
        return losses

    
class AggregateModel(nnx.Module):
    def __init__(self, functional_subgraphs: List[nnx.Module], global_nl: Callable):
        self.functional_subgraphs = functional_subgraphs
        self.global_nl = global_nl
        self.constant = nnx.Param(jnp.array(0.0))
    
    # @nnx.jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = self.constant * jnp.ones((x.shape[0],))
        # NOTE: individual models will be in [0, 1] scale, but this
        # won't necessarily b/c sum. Could have it be, but shrug.
        for fs in self.functional_subgraphs:
            res += fs(x)
        return self.global_nl(res)
    
    def evaluate_batched(self, x: jnp.ndarray) -> jnp.ndarray:
        res = self.constant.value * jnp.ones((x.shape[0],))
        for fs in self.functional_subgraphs:
            res += fs.evaluate_batched(x)
        return self.global_nl(res)
    
    def evaluate_seq(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the function sequentially."""
        res = self.constant.value * jnp.ones((x.shape[0],))
        for fs in self.functional_subgraphs:
            res += fs.evaluate_seq(x)
        return self.global_nl(res)