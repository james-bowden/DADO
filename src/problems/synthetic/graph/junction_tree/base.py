import numpy as np
from typing import Optional

from src.decomposition.graphs import JunctionTree
from src.problems.synthetic.graph.base import SyntheticGraphObjective


class SyntheticJunctionTreeObjective(SyntheticGraphObjective):
    """
    Synthetic objective function based on a functional tree.
    The function value is computed by evaluating a functional tree.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a synthetic tree objective function.
        
        Args:
            D: Dimension of each position in the sequence
            L: Length of sequence (# cliques in junction tree)
            fg: Functional tree that defines the objective
            maximize: Whether to maximize (True) or minimize (False) the objective
            best_x: Global optimum of the objective (if known)
        """
        super().__init__(*args, **kwargs)
        self.obj_name = "SyntheticJunctionTreeObjective"
        # check is tree
        assert isinstance(self.fg, JunctionTree), "FunctionalGraph must be a tree."
