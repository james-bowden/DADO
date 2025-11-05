import jax.numpy as jnp
import numpy as np

from src.decomposition.fgm.tabular.base import TabularFunctionalTree
from src.problems.synthetic.graph.tree.base import SyntheticTreeObjective


class SE_TreeObjective1(SyntheticTreeObjective):
    """
    Implementation of a specific tree objective with L=2, D=20.
    All scores <= 0, maximum at 0.
    Explicitly encodes sign epistasis on the edge (0,1).
    """

    def __init__(self, *args, **kwargs):
        # Parameters
        L = 2
        D = 20
        np.random.seed(42)

        # Node functions: negative values, with state 0 best (=0)
        node_functions = [jnp.zeros(D) for _ in range(L)]
        # for _ in range(L):
        #     arr = -1. * np.abs(np.random.randn(D))
        #     arr[0] = 0.0  # reference state
        #     node_functions.append(jnp.array(arr))

        # Edge structure: single edge (0,1)
        edges = [(0, 1)]

        # Edge function with explicit sign epistasis
        ef = np.zeros((D, D))

        # Design sign epistasis:
        # Mutation at node0 (0->1) is beneficial if node1=0,
        # deleterious if node1=1.
        ef[0, 0] = 0.0     # baseline optimum
        ef[1, 0] = +4.0    # relatively better in background node1=0
        ef[0, 1] = 0.0
        ef[1, 1] = -4.0    # relatively worse in background node1=1

        # Fill other entries with random negatives
        for i in range(D):
            for j in range(D):
                if (i, j) not in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                    ef[i, j] = -2.0 * np.abs(np.random.randn())

        # Shift so the maximum entry = 0, everything else <= 0
        ef = ef - np.max(ef)

        edge_functions = {
            edges[0]: jnp.array(ef),
        }

        # Build functional tree
        fg = TabularFunctionalTree(
            n_nodes=L,
            n_states_list=[D] * L,
            edges=edges,
            node_functions=node_functions,
            edge_functions=edge_functions,
            root=0,
        )

        # Initialize parent class
        super().__init__(D=[D] * L, fg=fg, *args, **kwargs)
        self.obj_name = "SE_TreeObjective1"

        if self.verbose:
            print(f"{self.obj_name}: {L} variables with {D} states each")