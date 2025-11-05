import numpy as np
import jax
import jax.numpy as jnp
from typing import Union, List, Tuple, Optional, Dict, Callable
from copy import deepcopy
from functools import partial

import optax as ox
from flax import nnx

from src.decomposition.graphs import Graph, Tree, JunctionTree, GeneralizedGraph, DisconnectedGraph
from src.decomposition.fgm.tabular.utils import random_edge_functions, random_node_functions, zero_node_functions, zero_edge_functions

# NOTE: this file isn't generally used; use MLPFunctionalGraph instead.

class TabularFunctionalGraph(Graph, nnx.Module):
    """A graph where each edge has an associated function that takes two discrete inputs and each node has an associated function."""
    
    def __init__(self, 
                 n_nodes: int, 
                 edges: Union[List[Tuple[int, int]], np.ndarray],
                 edge_functions: Dict[Tuple[int, int], jnp.ndarray],
                 node_functions: List[jnp.ndarray],
                 n_states_list: List[int],
                 *args, 
                 use_constant: bool = True,
                 **kwargs):
        """
        Initialize a functional graph with n_nodes nodes and optional edges and edge functions.
        
        Args:
            n_nodes: Number of nodes in the graph
            edges: Optional edge list or adjacency matrix
            edge_functions: Optional dictionary mapping edges to functions that take two integers and return a float
            node_functions: Optional dictionary mapping nodes to arrays of length n_states, where node_functions[i][s] 
                           gives the value for node i in state s
            n_states: Number of possible states for each node (default: 2)
        """
        super().__init__(n_nodes, edges, *args, **kwargs)
        self.n_states_list = jnp.array(n_states_list)
        assert len(n_states_list) == n_nodes, f"Number of n_states_list {len(n_states_list)} does not match number of nodes {n_nodes}"
        
        # Validate edge functions
        assert len(edge_functions) == len(edges), f"Number of edge functions {len(edge_functions)} does not match number of edges {len(edges)}"
        for edge, func in edge_functions.items():
            if edge not in edges and (edge[::-1] not in edges): # check both directions
                raise ValueError(f"Edge {edge} has a function but is not in the graph")
            assert func.shape == (n_states_list[edge[0]], n_states_list[edge[1]]), f"Edge function for {edge} has shape {func.shape}, but n_states_list is {n_states_list}"
        
        # Validate node functions
        assert len(node_functions) == n_nodes, f"Number of node functions {len(node_functions)} does not match number of nodes {n_nodes}"
        for node, func in enumerate(node_functions):
            assert func.shape == (n_states_list[node],), f"Node function for node {node} has shape {func.shape}, but n_states_list is {n_states_list}"
        
        # Convert node functions to nnx.Param (stack into array)
        self.node_functions = [nnx.Param(jnp.array(node_func)) for node_func in node_functions]
        
        # Convert edge functions to nnx.Param (need to handle variable shapes)
        self.edge_functions = {
            edge: nnx.Param(jnp.array(edge_func)) for edge, edge_func in edge_functions.items()
        }
        
        # Store original functions for reference
        self.original_edge_functions = deepcopy(edge_functions)
        self.original_node_functions = deepcopy(node_functions)

        self.use_constant = use_constant
        self.constant = nnx.Param(jnp.array(0.0))
        self.global_nl = lambda x: x
        
        # in case of exp weight fn
        self.eval_op = jnp.add

        self.prepare_evaluate_jax()

    @classmethod
    def random(cls, n_nodes: int, n_edges: int, n_states_list: List[int], seed: int = 48, use_constant: bool = True) -> 'TabularFunctionalGraph':
        """
        Create a random functional graph with n_nodes nodes and n_edges edges.
        Edge functions are randomly generated with Gaussian coefficients.
        
        Args:
            n_nodes: Number of nodes
            n_edges: Number of edges to sample
            n_states: Number of possible states for each node
        """
        graph = Graph.random(n_nodes, n_edges)
        edge_functions = random_edge_functions(graph.get_edge_list(), n_states_list)
        node_functions = random_node_functions(n_states_list)
        
        return cls(n_nodes, graph.get_edge_list(), edge_functions, node_functions, n_states_list, use_constant=use_constant)
    
    @classmethod
    def fixed_graph_random(cls, graph: Graph, n_states_list: List[int], seed: int = 48, use_constant: bool = True) -> 'TabularFunctionalGraph':
        """
        Create a random functional graph with n_nodes nodes and n_edges edges.
        Edge functions are randomly generated with Gaussian coefficients.
        """
        edge_functions = random_edge_functions(graph.edges, n_states_list)
        node_functions = random_node_functions(n_states_list)

        return cls(graph.n_nodes, graph.edges, edge_functions, node_functions, n_states_list, use_constant=use_constant)

    def apply_weight_fn(self, weight_fn_name: str):
        match weight_fn_name:
            case 'id' | 'identity' | 'none' | '':
                pass
            case 'fg_pos_scale':
                self.w_make_positive_scale()
            case 'fg_pos':
                self.w_make_positive()
            case 'fg_exp_scale':
                self.w_exponentiate_scale()
            case 'fg_exp':
                self.w_exponentiate()
            case _:
                raise ValueError(f"Invalid weight function: {weight_fn_name}")

    def w_make_positive(self):
        """
        Make all node, edge functions positive by setting a baseline.
        """
        baseline = 0 # must be non-negative
        for node_func in self.node_functions:
            baseline = min(baseline, jnp.min(node_func.value))
        for edge_func in self.edge_functions.values():
            baseline = min(baseline, jnp.min(edge_func))

        for node_func in self.node_functions:
            node_func.value = node_func.value - baseline
        for edge, edge_func in self.edge_functions.items():
            self.edge_functions[edge].value = edge_func.value - baseline
        self.prepare_evaluate_jax()

    # NOTE: global shift shouldn't affect sign epistasis. 
    # If you shift node/edge functions individually though, it will.
    def w_make_positive_scale(self):
        self.w_make_positive()
        max_value = -1
        for node_func in self.node_functions:
            max_value = max(max_value, jnp.max(node_func.value))
        for edge_func in self.edge_functions.values():
            max_value = max(max_value, jnp.max(edge_func))
        
        for node_func in self.node_functions:
            node_func.value = node_func.value / max_value
        for edge, edge_func in self.edge_functions.items():
            self.edge_functions[edge].value = edge_func.value / max_value
        
        self.prepare_evaluate_jax()

    def w_exponentiate(self):
        """
        Exponentiate all node, edge functions.
        """
        for node_func in self.node_functions:
            node_func.value = jnp.exp(node_func.value)
        for edge_func in self.edge_functions.values():
            edge_func.value = jnp.exp(edge_func.value)
        
        self.eval_op = jnp.multiply
        self.prepare_evaluate_jax()

    def w_exponentiate_scale(self):
        self.w_exponentiate()
        max_value = -1
        for node_func in self.node_functions:
            max_value = max(max_value, jnp.max(node_func.value))
        for edge_func in self.edge_functions.values():
            max_value = max(max_value, jnp.max(edge_func.value))
        
        for node_func in self.node_functions:
            node_func.value = node_func.value / max_value
        for edge_func in self.edge_functions.values():
            edge_func.value = edge_func.value / max_value
        
        self.prepare_evaluate_jax()
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.jit_evaluate(x)

    @staticmethod
    def evaluate_jax_single(x: jnp.ndarray,
                           node_functions: List,
                           edge_functions: Dict,
                           edges: List[Tuple[int, int]],
                           nodes: List[int],
                           eval_op: Callable) -> jnp.ndarray:
        total = jax.lax.select(eval_op == jnp.add, 0.0, 1.0)
        
        for node in nodes:
            total = eval_op(total, node_functions[node].value[x[node]])
        
        for edge in edges:
            node_i, node_j = edge
            total = eval_op(total, edge_functions[edge].value[x[node_i], x[node_j]])
        
        return total

    def prepare_evaluate_jax(self):
        """
        Set self.evaluate to jit, vmap version.
        """
        c = 0.0
        if self.use_constant:
            c = self.constant.value
        self.jit_evaluate = nnx.jit( # NOTE: these were jax, but I think need to backprop through this.
            nnx.vmap(
                lambda x: self.global_nl(
                    c + self.evaluate_jax_single(
                        x, self.node_functions, self.edge_functions, 
                        self.edges, self.graph.nodes(), self.eval_op
        ))))

    # @nnx.jit
    def node_function(self, node: int, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the node function for a single input."""
        node_state = x[node]
        return self.node_functions[node].value[node_state]
        
    # @nnx.jit
    def edge_function(self, edge: Tuple[int, int], x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the edge function for a single input."""
        return self.edge_functions[edge].value[x[edge[0]], x[edge[1]]]

    @staticmethod # NOTE: important!! avoid tracing / mutation.
    def _split_merge(fg: 'TabularFunctionalGraph') -> jnp.ndarray:
        graphdef, state = nnx.split(fg)
        new_fg = nnx.merge(graphdef, state)
        return new_fg
    
    @staticmethod
    def fit(
        fg: 'TabularFunctionalGraph', 
        x: jnp.ndarray, y: jnp.ndarray,
        n_iter_inner: int = 20,
        n_epochs: int = 100,
        optimizer = ox.adam(1e-3),
        weight_fn_name: str = 'fg_pos_scale',
        seed: int = 48,
    ):
        """
        Fit the tabular functional graph to the given values.
        """
        nnx_optimizer = nnx.Optimizer(fg, optimizer)
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
            
            # Use full dataset for now (can add batching later)
            x_batch = x
            y_batch = y
            
            (model, nnx_opt, _, _), losses = scan_inner(
                (model, nnx_opt, x_batch, y_batch), 
                jnp.arange(n_iter_inner)
            )
            return (model, nnx_opt, key), losses[-1]

        fg.train()
        (fg, nnx_optimizer, _), losses = nnx.scan(f=epoch_step, length=n_epochs)(
            (fg, nnx_optimizer, key), jnp.arange(n_epochs)
        )
        fg.eval()
        
        fg.apply_weight_fn(weight_fn_name)
        return losses

class TabularFunctionalTree(TabularFunctionalGraph, Tree):
    """A tree where each edge has an associated function that takes two discrete inputs."""

    def __init__(self, n_nodes: int, edges: Union[List[Tuple[int, int]], np.ndarray],
                 edge_functions: Dict[Tuple[int, int], jnp.ndarray],
                 node_functions: jnp.ndarray,
                 n_states_list: List[int],
                 root: Optional[int] = None,
                 original_nodes: Optional[List[set]] = None,
                 use_constant: bool = True
        ):
        """
        Initialize a functional tree with n_nodes nodes and optional edges, edge functions, and root.
        
        Args:
            n_nodes: Number of nodes in the tree
            edges: Optional edge list or adjacency matrix
            edge_functions: Optional dictionary mapping edges to functions that take two integers and return a float
            node_functions: Optional dictionary mapping nodes to arrays of length n_states
            n_states: Number of possible states for each node (default: 2)
            root: Optional root node index
            original_nodes: Optional list of sets representing the original graph nodes in each junction tree node
        """
        # Initialize both parent classes
        Tree.__init__(self, n_nodes, edges, root)
        TabularFunctionalGraph.__init__(self, n_nodes, edges, edge_functions, node_functions, n_states_list, use_constant=use_constant)
        self.original_nodes = original_nodes
    
    @classmethod
    def random(cls, n_nodes: int, n_states_list: List[int], seed: int = 48, root: Optional[int] = None, use_constant: bool = True) -> 'TabularFunctionalTree':
        """
        Create a random functional tree with n_nodes nodes.
        Edge functions are randomly generated with Gaussian coefficients.
        
        Args:
            n_nodes: Number of nodes
            n_states: Number of possible states for each node
            root: Optional root node index
        """
        tree = Tree.random(n_nodes, root=root)
        edge_functions = random_edge_functions(tree.get_edge_list(), n_states_list)
        node_functions = random_node_functions(n_states_list)
        
        return cls(n_nodes, tree.get_edge_list(), edge_functions, node_functions, n_states_list, tree.root, use_constant=use_constant)
    
    @classmethod
    def fixed_graph_random(
        cls, graph: Tree, n_states_list: List[int], 
        root: Optional[int] = None, seed: int = 48, 
        use_constant: bool = True) -> 'TabularFunctionalTree':
        """
        Create a random functional graph with n_nodes nodes and n_edges edges.
        Edge functions are randomly generated with Gaussian coefficients.
        """
        edge_functions = random_edge_functions(graph.edges, n_states_list)
        node_functions = random_node_functions(n_states_list)

        return cls(graph.n_nodes, graph.edges, edge_functions, node_functions, n_states_list, root, use_constant=use_constant)

    def message_passing(self):
        """
        Perform message passing on the tree.
        """
        max_D = max(self.n_states_list)
        V = np.array(self.node_functions)
        Q = np.zeros((self.n_nodes, max_D, max_D)) # (node, parent_state, node_state)
        A = np.zeros_like(Q)
        for node in self.node_order[::-1]: # pass messages up tree
            parent = self.parents[node]
            for child in self.children[node]: # used to choose node
                V[node] += Q[child].max(axis=-1) # max over node states
            if parent != -1: # used to choose node (child of edge)
                Q[node, :, :] = V[node] + self.edge_functions[(parent, node)]
                A[node, :, :] = Q[node] - Q[node].mean(axis=-1)[:, None] # mean over node states

        return V, Q, A

class TabularFunctionalJunctionTree(TabularFunctionalGraph, JunctionTree):
    """A junction tree where each node is a set of nodes from the original graph."""
    def __init__(self, n_nodes: int, edges: Union[List[Tuple[int, int]], np.ndarray],
                 index_to_nodes: List[set], nodes_to_index: Dict[set, int],
                 node_functions: List[jnp.ndarray],
                 edge_functions: Dict[Tuple[int, int], jnp.ndarray],
                 n_states_list: List[int],
                 jt_n_states_list: List[int],
                 root: Optional[int] = None,
                 use_constant: bool = True):
        TabularFunctionalGraph.__init__(
            self, n_nodes, edges, 
            edge_functions, node_functions, jt_n_states_list,
            index_to_nodes=index_to_nodes, nodes_to_index=nodes_to_index, root=root, use_constant=use_constant
        )
        self.original_n_states_list = n_states_list

    @classmethod
    def random(cls, n_nodes: int, n_edges: int, seed: int = 48, n_states_list: List[int] = None) -> 'TabularFunctionalJunctionTree':
        raise NotImplementedError("FunctionalJunctionTree.random not implemented")

    @classmethod
    def fixed_graph_random(cls, 
                           jt: JunctionTree,
                           n_states_list: List[int], 
                           seed: int = 48,
                           use_constant: bool = True
        ) -> 'TabularFunctionalJunctionTree':
        jt_n_states_list = [
            int(jnp.prod(jnp.array([n_states_list[var] for var in jt.index_to_nodes[i]])))
            for i in range(jt.n_nodes)
        ]
        node_functions = random_node_functions(jt_n_states_list)
        g = jt.graph
        edge_functions = random_edge_functions(g.edges, jt_n_states_list)
        return cls(
            jt.n_nodes, g.edges, 
            jt.index_to_nodes, jt.nodes_to_index, 
            node_functions, edge_functions, 
            n_states_list, jt_n_states_list, 
            jt.root,
            use_constant=use_constant
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.jit_evaluate(x)

    @staticmethod
    def evaluate_jax_single(x: jnp.ndarray,
                           node_functions: List,
                           edge_functions: Dict,
                           edges: List[Tuple[int, int]],
                           nodes: List[int],
                           index_to_nodes: List[set],
                           original_n_states_list: jnp.ndarray,
                           eval_op: Callable) -> jnp.ndarray:
        total = jax.lax.select(eval_op == jnp.add, 0.0, 1.0)

        # NOTE: overloaded b/c need to index x by sets of nodes using mixed-radix encoding
        for node in nodes:
            clique = index_to_nodes[node]
            clique_list = sorted(list(clique))  # Convert set to sorted list for consistency
            
            # Get states for variables in this clique
            clique_states = jnp.array([x[var] for var in clique_list])
            
            # Get number of states for each variable in the clique
            clique_n_states = jnp.array([original_n_states_list[var] for var in clique_list])
            
            # Convert to joint state index using mixed-radix encoding
            radix_multipliers = jnp.cumprod(jnp.concatenate([jnp.array([1]), clique_n_states[:-1]]))
            joint_state_idx = jnp.sum(clique_states * radix_multipliers)
            
            total = eval_op(total, node_functions[node].value[joint_state_idx])
        
        for edge in edges:
            i, j = edge
            # Same logic for edge cliques
            i_clique = index_to_nodes[i]
            j_clique = index_to_nodes[j]
            i_clique_list = sorted(list(i_clique))
            j_clique_list = sorted(list(j_clique))
            
            # Get joint state indices for both cliques
            i_clique_states = jnp.array([x[var] for var in i_clique_list])
            j_clique_states = jnp.array([x[var] for var in j_clique_list])
            
            i_clique_n_states = jnp.array([original_n_states_list[var] for var in i_clique_list])
            j_clique_n_states = jnp.array([original_n_states_list[var] for var in j_clique_list])
            
            i_radix_multipliers = jnp.cumprod(jnp.concatenate([jnp.array([1]), i_clique_n_states[:-1]]))
            j_radix_multipliers = jnp.cumprod(jnp.concatenate([jnp.array([1]), j_clique_n_states[:-1]]))
            
            i_joint_state_idx = jnp.sum(i_clique_states * i_radix_multipliers)
            j_joint_state_idx = jnp.sum(j_clique_states * j_radix_multipliers)
            
            total = eval_op(total, edge_functions[edge].value[i_joint_state_idx, j_joint_state_idx])
        
        return total

    def prepare_evaluate_jax(self):
        """
        Set self.evaluate to jit, vmap version.
        """
        c = 0.0
        if self.use_constant:
            c = self.constant.value
        self.jit_evaluate = nnx.jit( # NOTE: these were jax, but I think need to backprop through this.
            nnx.vmap(
                lambda x: self.global_nl(
                    c + self.evaluate_jax_single(
                        x, self.node_functions, self.edge_functions, 
                        self.graph.edges(), self.graph.nodes(), 
                        self.index_to_nodes, 
                        self.original_n_states_list,
                        self.eval_op,
        ))))

    # @nnx.jit
    def node_function(self, node: int, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the node function for a single input.
        Node is the clique index."""
        clique = self.index_to_nodes[node]
        clique_list = sorted(list(clique))
        node_state = x[jnp.array(clique_list)] # TODO: is this correct?
        clique_n_states = jnp.array([self.original_n_states_list[var] for var in clique_list])
        radix_multipliers = jnp.cumprod(jnp.concatenate([jnp.array([1]), clique_n_states[:-1]]))
        joint_state_idx = jnp.sum(node_state * radix_multipliers)
        return self.node_functions[node].value[joint_state_idx]
        
    # @nnx.jit
    def edge_function(self, edge: Tuple[int, int], x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the edge function for a single input.
        Edge is tuple of clique indices."""
        i, j = edge
        i_clique = self.index_to_nodes[i]
        j_clique = self.index_to_nodes[j]
        i_clique_list = sorted(list(i_clique))
        j_clique_list = sorted(list(j_clique))
        edge_state = (x[jnp.array(i_clique_list)], x[jnp.array(j_clique_list)])
        i_clique_n_states = jnp.array([self.original_n_states_list[var] for var in i_clique_list])
        j_clique_n_states = jnp.array([self.original_n_states_list[var] for var in j_clique_list])
        i_radix_multipliers = jnp.cumprod(jnp.concatenate([jnp.array([1]), i_clique_n_states[:-1]]))
        j_radix_multipliers = jnp.cumprod(jnp.concatenate([jnp.array([1]), j_clique_n_states[:-1]]))
        i_joint_state_idx = jnp.sum(edge_state[0] * i_radix_multipliers)
        j_joint_state_idx = jnp.sum(edge_state[1] * j_radix_multipliers)
        return self.edge_functions[edge].value[i_joint_state_idx, j_joint_state_idx]
    
    @staticmethod
    def fit(
        fg: 'TabularFunctionalJunctionTree', 
        x: jnp.ndarray, y: jnp.ndarray,
        n_iter_inner: int = 20,
        n_epochs: int = 100,
        optimizer = ox.adam(1e-3),
        weight_fn_name: str = 'fg_pos_scale',
        seed: int = 48,
    ):
        """
        Fit the tabular functional junction tree to the given values.
        """
        nnx_optimizer = nnx.Optimizer(fg, optimizer)
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
            
            # Use full dataset for now (can add batching later)
            x_batch = x
            y_batch = y
            
            (model, nnx_opt, _, _), losses = scan_inner(
                (model, nnx_opt, x_batch, y_batch), 
                jnp.arange(n_iter_inner)
            )
            return (model, nnx_opt, key), losses[-1]

        fg.train()
        (fg, nnx_optimizer, _), losses = nnx.scan(f=epoch_step, length=n_epochs)(
            (fg, nnx_optimizer, key), jnp.arange(n_epochs)
        )
        fg.eval()
        
        fg.apply_weight_fn(weight_fn_name)
        return losses
    
class TabularGeneralizedFunctionalGraph(GeneralizedGraph, nnx.Module):
    """A functional graph that automatically decomposes itself into appropriate subcomponents."""
    
    def __init__(self, 
                 generalized_graph: GeneralizedGraph,
                 functional_subgraphs: List[TabularFunctionalGraph],
                 *args, **kwargs):
        """Initialize a generalized functional graph from a generalized graph and functional subgraphs."""
        # Copy structure from generalized graph
        super().__init__(generalized_graph.n_nodes, generalized_graph.get_edge_list(), *args, verbose=False, **kwargs)
        self.subgraphs = generalized_graph.subgraphs
        self.node_to_subgraph = generalized_graph.node_to_subgraph
        self.subgraph_node_indices = generalized_graph.subgraph_node_indices
        
        # Store functional subgraphs
        self.functional_subgraphs = functional_subgraphs

        self.constant = nnx.Param(jnp.array(0.0)) # always use constant for GGFGM.
        self.global_nl = nnx.sigmoid # lambda x: x
    
    @classmethod
    def fixed_graph_random(cls, 
                          graph: GeneralizedGraph,
                          n_states_list: List[int],
                          seed: int = 48) -> 'TabularGeneralizedFunctionalGraph':
        """Create a generalized functional graph with random functions for a fixed graph structure."""
        # Create functional subgraphs for each component
        functional_subgraphs = []
        
        for subgraph_idx, subgraph in enumerate(graph.subgraphs):
            global_node_indices = graph.subgraph_node_indices[subgraph_idx]
            sub_n_states_list = [n_states_list[i] for i in global_node_indices]
            
            # Create the appropriate functional subgraph using fixed_graph_random
            if isinstance(subgraph, DisconnectedGraph):
                functional_subgraph = TabularFunctionalGraph.fixed_graph_random(
                    subgraph, sub_n_states_list, seed + subgraph_idx, 
                    use_constant=False, # no constants for subgraphs
                )
            elif isinstance(subgraph, JunctionTree):
                functional_subgraph = TabularFunctionalJunctionTree.fixed_graph_random(
                    subgraph, sub_n_states_list, seed + subgraph_idx,
                    use_constant=False, # no constants for subgraphs
                )
            elif isinstance(subgraph, Tree):
                functional_subgraph = TabularFunctionalTree.fixed_graph_random(
                    subgraph, sub_n_states_list, None, seed + subgraph_idx, # no root
                    use_constant=False, # no constants for subgraphs
                )
            else:
                raise ValueError(f"Subgraph was neither DisconnectedGraph, Tree, nor JunctionTree. "
                               f"Subgraph type: {type(subgraph)}, edges: {subgraph.get_edge_list()}")
            
            functional_subgraphs.append(functional_subgraph)
        
        return cls(graph, functional_subgraphs)
    
    def apply_weight_fn(self, weight_fn_name: str):
        for subgraph in self.functional_subgraphs:
            subgraph.apply_weight_fn(weight_fn_name)
    
    # @partial(jax.vmap, in_axes=(None, 0))
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the function by combining evaluations from all subgraphs."""
        total = jnp.zeros(x.shape[0])
        
        for subgraph_idx, functional_subgraph in enumerate(self.functional_subgraphs):
            global_node_indices = self.subgraph_node_indices[subgraph_idx]
            sub_x = x[:, global_node_indices]
            sub_values = functional_subgraph(sub_x)
            total += sub_values
        
        # return self.global_nl(total + self.constant.value)
        return self.global_nl(total) + self.constant.value
    
    @staticmethod
    def fit(
        fg: 'TabularGeneralizedFunctionalGraph',
        x: jnp.ndarray, y: jnp.ndarray,
        n_iter_inner: int = 20,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_fn_name: str = 'fg_pos_scale',
        seed: int = 48,
    ):
        """
        Fit the tabular generalized functional graph to the given values.
        """
        optimizer = ox.adamw(learning_rate)
        nnx_optimizer = nnx.Optimizer(fg, optimizer)
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
            
            # Use full dataset for now (can add batching later)
            x_batch = x
            y_batch = y
            
            (model, nnx_opt, _, _), losses = scan_inner(
                (model, nnx_opt, x_batch, y_batch), 
                jnp.arange(n_iter_inner)
            )
            return (model, nnx_opt, key), losses[-1]

        fg.train()
        (fg, nnx_optimizer, _), losses = nnx.scan(f=epoch_step, length=n_epochs)(
            (fg, nnx_optimizer, key), jnp.arange(n_epochs)
        )
        fg.eval()
        
        fg.apply_weight_fn(weight_fn_name)
        return losses

class TabularFunctionalHypergraph(Graph):
    pass