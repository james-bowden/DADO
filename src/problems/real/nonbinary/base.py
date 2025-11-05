from src.problems.real.base import RealObjective


class NonBinaryObjective(RealObjective):
    """Base class for binary objective functions."""
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a binary objective function.
        
        Args:
            D: Dimension of each position in the sequence
            L: Length of sequence
            maximize: Whether to maximize (True) or minimize (False) the objective
            best_x: Global optimum of the objective (if known)
        """
        super().__init__(*args, **kwargs)
        for D in self.D:
            assert D >= 2, f"Non-binary objective functions must have D>=2, found D={self.D}"