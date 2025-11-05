import networkx as nx
from networkx.classes.reportviews import OutEdgeView, EdgeView
import numpy as np
from typing import Union, List, Tuple, Optional, Dict
import random
import matplotlib.pyplot as plt
import math

def is_clique(G, nodes):
    """Check if the given set of nodes forms a clique in G."""
    subgraph = G.subgraph(nodes)
    n = len(subgraph)
    return subgraph.number_of_edges() == n * (n - 1) // 2

def verify_junction_tree(G, JT):
    """Verify that JT is a valid junction tree for graph G."""
    # 1. All junction tree nodes should be cliques in triangulated G
    if not nx.is_chordal(G):
        G, _ = nx.complete_to_chordal_graph(G)
    
    for clique in JT.nodes:
        if not is_clique(G, clique):
            print(f"Invalid: {clique} is not a clique in triangulated G")
            return False

    # 2. Running intersection property: for all pairs, the separator must appear on the path between them
    cliques = list(JT.nodes)
    for i in range(len(cliques)):
        for j in range(i+1, len(cliques)):
            intersection = set(cliques[i]).intersection(cliques[j])
            if not intersection:
                continue
            try:
                for node in nx.shortest_path(JT, cliques[i], cliques[j]):
                    if not intersection.issubset(node):
                        print(f"Invalid running intersection property between {cliques[i]} and {cliques[j]}")
                        return False
            except nx.NetworkXNoPath:
                print(f"No path between {cliques[i]} and {cliques[j]}")
                return False

    # 3. All variables in G must be covered by cliques
    covered = set().union(*JT.nodes)
    if set(G.nodes) != covered:
        print("Invalid: some variables are missing from junction tree cliques")
        return False

    return True

def prune_redundant_cliques(JT):
    """Remove cliques that are strict subsets of others."""
    cliques = list(JT.nodes)
    keep = set()

    # Brute force: only keep cliques that are not strict subsets of any other
    for i, c1 in enumerate(cliques):
        if any(set(c1) < set(c2) for j, c2 in enumerate(cliques) if i != j):
            continue
        keep.add(c1)

    # Rebuild pruned junction tree by connecting retained cliques by their separator size
    pruned = nx.Graph()
    pruned.add_nodes_from(keep)

    for c1 in keep:
        for c2 in keep:
            if c1 != c2:
                sep = set(c1).intersection(set(c2))
                if sep:
                    pruned.add_edge(c1, c2, weight=len(sep))

    # Build maximum weight spanning tree on clique overlap
    if pruned.number_of_edges() > 0:
        pruned = nx.maximum_spanning_tree(pruned, weight="weight")

    return pruned

class Graph:
    """A class for general graphs built on top of NetworkX."""
    
    def __init__(self, 
                 n_nodes: int, 
                 edges: Union[List[Tuple[int, int]], np.ndarray],
                 directed: bool = False):
        """
        Initialize a graph with n_nodes nodes and optional edges.
        
        Args:
            n_nodes: Number of nodes in the graph
            edges: Optional edge list or adjacency matrix. If edge list, should be a list of tuples (i,j).
                  If adjacency matrix, should be a numpy array of shape (n_nodes, n_nodes).
        """
        self.n_nodes = n_nodes
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self.graph.add_nodes_from(range(n_nodes))
        
        if edges is not None:
            if isinstance(edges, (list, OutEdgeView, EdgeView)):
                self.graph.add_edges_from(edges)
            elif isinstance(edges, np.ndarray):
                if edges.shape != (n_nodes, n_nodes):
                    raise ValueError("Adjacency matrix must be square with shape (n_nodes, n_nodes)")
                for i in range(n_nodes):
                    for j in range(i+1, n_nodes):
                        if edges[i,j] == 1:
                            if i < j:
                                self.graph.add_edge(i, j)
                            else: # i == j case should never happen.
                                self.graph.add_edge(j, i)
            else:
                raise ValueError(f"Invalid edges type: {type(edges)}")
        
        self.n_edges = len(self.graph.edges())
        self.edges = [tuple(sorted(e)) for e in self.graph.edges()]
    
    @classmethod
    def random(cls, n_nodes: int, n_edges: int) -> 'Graph':
        """
        Create a random graph with n_nodes nodes and n_edges edges.
        Edges are sampled uniformly from all possible edges.
        
        Args:
            n_nodes: Number of nodes
            n_edges: Number of edges to sample
        """
        if n_edges > n_nodes * (n_nodes - 1) // 2:
            raise ValueError("Too many edges requested for the number of nodes")
            
        possible_edges = [(i,j) for i in range(n_nodes) for j in range(i+1, n_nodes)]
        selected_edges = random.sample(possible_edges, n_edges)
        
        return cls(n_nodes, selected_edges)
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Return the adjacency matrix of the graph."""
        return nx.adjacency_matrix(self.graph).toarray().astype(bool)
    
    def get_edge_list(self) -> List[Tuple[int, int]]:
        """Return the edge list of the graph."""
        return list(self.graph.edges())
    
    def to_junction_tree(self, verbose: bool = False) -> 'Tree':
        """
        Convert the graph to a junction tree using NetworkX's junction tree algorithm.
        
        Returns:
            A Tree object representing the junction tree of the graph.
            The nodes of the junction tree are sets of nodes from the original graph.
        """
        # Use NetworkX's junction tree algorithm
        jt = nx.junction_tree(self.graph)
        n_nodes_before = len(jt.nodes())
        max_clique_size_before = max(len(c) for c in jt.nodes())
        jt = prune_redundant_cliques(jt)
        n_pruned = n_nodes_before - len(jt.nodes())
        max_clique_size_after = max(len(c) for c in jt.nodes())
        if verbose and n_pruned != 0: print(f"Pruned {n_pruned} nodes from JT!")
        if verbose: print(f"Max clique size before: {max_clique_size_before}, after: {max_clique_size_after}")
        # TODO: maybe also want to look at mean, median, distribution of clique size
        if not isinstance(self, Tree): # tree is always valid JT
            assert verify_junction_tree(self.graph, jt), "Pruned JT is not valid"
            
        if verbose:
            nx.draw(jt, with_labels=True)
            plt.show()
        
        # Convert to our Tree class
        n_nodes = len(jt.nodes())
        nodes_to_index = {node: idx for idx, node in enumerate(jt.nodes())}
        indexed_edges = [
            (nodes_to_index[u], nodes_to_index[v]) for u, v in jt.edges
        ]
        
        jt = JunctionTree(
            n_nodes, indexed_edges, 
            index_to_nodes=list(jt.nodes()), nodes_to_index=nodes_to_index)
        return jt

class Tree(Graph):
    """A class for trees built on top of NetworkX."""
    
    def __init__(self, 
                 n_nodes: int, 
                 edges: Union[List[Tuple[int, int]], np.ndarray], 
                 root: Optional[int] = None, 
                 directed: bool = True
                ):
        """
        Initialize a tree with n_nodes nodes and optional edges.
        
        Args:
            n_nodes: Number of nodes in the tree
            edges: Optional edge list or adjacency matrix. If edge list, should be a list of tuples (i,j).
                  If adjacency matrix, should be a numpy array of shape (n_nodes, n_nodes).
            root: Optional root node index. If not provided, a root will be chosen.
        """
        super().__init__(n_nodes, edges)
        if not nx.is_tree(self.graph):
            raise ValueError(f"The provided edges do not form a valid tree: {edges} / {self.graph.edges()}")
            
        # Set the root
        if root is not None:
            if root < 0 or root >= n_nodes:
                raise ValueError("Root node index out of range")
            self.root = root
        else:
            self.find_min_height_root()

        if directed:
            self.graph = self.to_directed_tree()

            self.node_order = list(nx.topological_sort(self.graph))            
            self.parents = []
            for n in range(self.n_nodes):
                pred = list(self.graph.predecessors(n))
                if len(pred) > 0:
                    self.parents.append(pred[0])
                else:
                    self.parents.append(-1)  # Use -1 instead of None for JAX compatibility
            self.children = [list(self.graph.successors(n)) for n in range(self.n_nodes)]

    
    @classmethod
    def random(cls, 
               n_nodes: int, 
               directed: bool = True,
               seed: int = 42,
               root: Optional[int] = None, 
            ) -> 'Tree':
        """
        Create a random tree with n_nodes nodes using NetworkX's random_tree function.
        
        Args:
            n_nodes: Number of nodes
            root: Optional root node index. If not provided, a root will be chosen from the tree.
        """
        if n_nodes == 0:
            return cls(0, [], None)
            
        # Generate a random tree using NetworkX
        nx_tree = nx.random_labeled_rooted_tree(n_nodes, seed=seed)
        edges = list(nx_tree.edges())

        return cls(n_nodes, edges, root, directed)

    def find_min_height_root(self) -> None:
        """
        Find and set the root that minimizes the tree height.
        Uses a center-finding algorithm based on repeatedly removing leaves.
        """
        if self.n_nodes == 0: return
        self.root = nx.center(self.graph)[0]
        
    def to_directed_tree(self) -> nx.DiGraph:
        """
        Convert the undirected tree to a directed tree with edges oriented away from the root.
        
        Returns:
            A NetworkX directed graph representing the tree with edges oriented away from the root.
        """
        if self.root is None:
            raise ValueError("Tree must have a root to create a directed tree")
            
        tree = nx.bfs_tree(self.graph, self.root)
        tree.root = self.root
        return tree

class JunctionTree(Tree):
    """A junction tree is a tree where each node is a set of nodes from the original graph.
    This means that edges should be between sets of nodes, not between nodes.
    """
    def __init__(
            self, n_nodes: int, edges: Union[List[Tuple[int, int]], np.ndarray],
            index_to_nodes: List[set], nodes_to_index: Dict[set, int],
            root: Optional[int] = None
        ):
        super().__init__(n_nodes, edges, root)
        self.index_to_nodes = index_to_nodes # where "index" is the clique index. # TODO: rename b/c cliques are edges.
        self.nodes_to_index = nodes_to_index
        self.n_cliques = len(index_to_nodes)
        
        # Calculate clique sizes including parent clique size (if it has one)
        self.clique_sizes, self.clique_parent_sizes = [], []
        for i in range(self.n_nodes):
            clique_size = len(index_to_nodes[i])
            parent_idx = self.parents[i]
            if parent_idx != -1:  # Has a parent
                parent_size = len(index_to_nodes[parent_idx])
                combined_size = clique_size + parent_size
            else:  # Root node, no parent
                combined_size = clique_size
            self.clique_sizes.append(clique_size)
            self.clique_parent_sizes.append(combined_size)

    @classmethod
    def random(cls, n_nodes: int, n_edges: int, seed: int = 48, n_states: int = 2) -> 'JunctionTree':
        raise NotImplementedError("JunctionTree.random not implemented")

class DisconnectedGraph(Graph):
    """A graph with no edges (all nodes are disconnected)."""
    
    def __init__(self, n_nodes: int):
        """Initialize a disconnected graph with n_nodes nodes and no edges."""
        super().__init__(n_nodes, [])
        self.edges = []
        
    @classmethod
    def random(cls, n_nodes: int) -> 'DisconnectedGraph':
        """Create a random disconnected graph with n_nodes nodes."""
        return cls(n_nodes)

class GeneralizedGraph(Graph):
    """A graph that automatically decomposes itself into subcomponents based on structure."""
    
    def __init__(self, n_nodes: int, edges: Union[List[Tuple[int, int]], np.ndarray], verbose: bool = True, node_size: int = 200):
        """Initialize a generalized graph and analyze its components."""
        super().__init__(n_nodes, edges)
        
        # Analyze graph structure and create subcomponents
        self.subgraphs = []
        self.node_to_subgraph = {}  # Maps global node index to (subgraph_index, local_node_index)
        self.subgraph_node_indices = []  # List of global node indices for each subgraph
        
        self._analyze_and_create_subgraphs(verbose, node_size)
    
    def _analyze_and_create_subgraphs(self, verbose: bool = True, node_size: int = 200):
        """Analyze the graph and create appropriate subgraph objects for each component."""
        # Get connected components
        components = list(nx.connected_components(self.graph))
        
        # Find isolated nodes (nodes with no edges)
        isolated_nodes = [node for node in self.graph.nodes() if self.graph.degree(node) == 0]
        isolated_component = set(isolated_nodes) if isolated_nodes else set()
        
        # Remove isolated component from other components if it exists
        if isolated_component:
            components = [comp for comp in components if not comp.issubset(isolated_component)]

        if verbose:
            total_components = len(components) + (1 if isolated_component else 0)
            ncols = math.ceil(math.sqrt(total_components))
            nrows = math.ceil(total_components / ncols)
            plot_idx = 0  # current subplot index
            fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))
            axes = np.array(axes).reshape(-1)
        
        # Handle isolated nodes with one DisconnectedGraph
        if isolated_component:
            isolated_indices = sorted(list(isolated_component))
            self.subgraph_node_indices.append(isolated_indices)
            
            # Create node mapping for isolated nodes
            for local_idx, global_idx in enumerate(isolated_indices):
                self.node_to_subgraph[global_idx] = (len(self.subgraphs), local_idx)
            
            subgraph = DisconnectedGraph(len(isolated_indices))

            if verbose:
                ax = axes[plot_idx]
                nx.draw(subgraph.graph, with_labels=True, node_color='orange', node_size=node_size, ax=ax)
                ax.set_title(f"{len(self.subgraphs)} - Edgeless")
                plot_idx += 1
            self.subgraphs.append(subgraph)
        
        # Handle each connected component
        for component in components:
            component_indices = sorted(list(component))
            self.subgraph_node_indices.append(component_indices)
            
            # Create node mapping for this component
            for local_idx, global_idx in enumerate(component_indices):
                self.node_to_subgraph[global_idx] = (len(self.subgraphs), local_idx)
            
            # Extract component subgraph
            component_graph = self.graph.subgraph(component)
            
            # Create local edge list (reindex to 0-based)
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(component_indices)}
            local_edges = [(global_to_local[u], global_to_local[v]) for u, v in component_graph.edges()]
            
            # Determine the appropriate subgraph type
            if nx.is_tree(component_graph):
                subgraph = Tree(len(component_indices), local_edges)
            else:
                subgraph = Graph(len(component_indices), local_edges).to_junction_tree()
                
            if verbose:
                ax = axes[plot_idx]
                color = None if isinstance(subgraph, JunctionTree) else 'lightgreen'
                nx.draw(subgraph.graph, with_labels=True, node_color=color, node_size=node_size, ax=ax)
                ax.set_title(f"{len(self.subgraphs)} - {'JT' if isinstance(subgraph, JunctionTree) else 'Tree'}")
                plot_idx += 1
            self.subgraphs.append(subgraph)
            
        if verbose:
            # Hide any unused subplots
            for j in range(plot_idx, len(axes)):
                axes[j].axis('off')
            plt.tight_layout(); plt.show()
            plt.clf()
        if len(self.subgraphs) > 1:
            print(f"{len(self.subgraphs)} subgraphs")
