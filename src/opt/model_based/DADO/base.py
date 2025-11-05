import jax
import jax.numpy as jnp
from typing import Tuple, Union, Optional

from src.opt.model_based.EDA.base import EDA


# TODO: this could just decide on tree or independent based on graph structure.
class DADO(EDA):
    """EDA using function decomposition and tree-structured density models."""
    
    def __init__(self, 
                *args, 
                use_transformer: bool = False,
                num_layers: Optional[int] = 4,
                dim_ff: Optional[int] = 64,
                num_heads: Optional[int] = 4,
                dim_attn: Optional[int] = 64,
                replay_buffer: Optional[bool] = False,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.num_heads = num_heads
        self.dim_attn = dim_attn
        self.hidden_dims = [dim_ff]*num_layers
        self.model = self.policy = None
        self.fg = self.objective.fg
        self.use_transformer = use_transformer
        self.replay_buffer = replay_buffer
        if not use_transformer:
            self.arc_string = "-".join(map(str, [dim_ff]*num_layers))
        else:
            self.arc_string = f"TF-{num_heads}H-{num_layers}L-{dim_attn}A-{dim_ff}FF"

    def get_solutions(self, n_points: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample from the trained density model.
        
        Returns:
            Tuple containing:
                - Array of shape (n_points, n_variables) containing the generated solutions
                - Array of shape (n_points,) containing the objective values for each solution
        """
        if self.policy is None:
            raise ValueError("Must call precompute() before get_solutions()")
        
        self.policy.eval() # make sure dropout is off
        
        self.rng, crng = jax.random.split(self.rng)
        samples, _ = self.policy.sample(jax.random.split(crng, n_points))
        objective_values = self.objective.query(samples)
        return samples, objective_values
