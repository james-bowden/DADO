import numpy as np
from typing import Optional

from src.problems.synthetic.graph.base import SyntheticGraphObjective


class SyntheticAdditiveObjective(SyntheticGraphObjective):
    """
    Synthetic objective function based on a functional additive tree.
    The function value is computed by evaluating a functional additive tree.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a synthetic additive objective function.
        
        Args:
            D: Dimension of each position in the sequence
            L: Length of sequence
            fg: Functional additive tree that defines the objective
            maximize: Whether to maximize (True) or minimize (False) the objective
            best_x: Global optimum of the objective (if known)
        """
        # Call parent class constructor with the FunctionalGraph
        super().__init__(*args, **kwargs)
        self.obj_name = "SyntheticAdditiveObjective"
        assert self.fg.n_edges == 0, "FunctionalGraph must decompose completely additively."
