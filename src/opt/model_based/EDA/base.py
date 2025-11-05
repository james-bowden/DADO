import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import numpy as np
from typing import Tuple
import os
import shutil

from src.opt.model_based.base import ModelBasedOptimizer


class EDA(ModelBasedOptimizer):
    """Base class for Estimation of Distribution Algorithms (EDAs)."""
    
    def __init__(self, 
                 *args,
                 num_epochs: int = 100,
                 num_inner_iter: int = 100,
                 num_MC_samples: int = 1000,
                 temp_Q: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = num_epochs
        self.num_inner_iter = num_inner_iter
        self.num_MC_samples = num_MC_samples
        self.policy = None
        self.temp_Q = temp_Q
        # Initialize RNG from seed set in parent Optimizer class
        if self.seed is not None:
            self.rng = jax.random.PRNGKey(self.seed)
        else:
            self.rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
    
    def precompute(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Train the density model using gradient-based optimization."""
        raise NotImplementedError("EDA subclasses must implement precompute()")

    def save(self, dictionary: dict = None):
        """Save the density model."""
        if self.policy is None:
            raise ValueError("Must call precompute() before save()")
        path = f"{os.environ['HOME']}/function-decomposed_EDA/checkpoints/policy/{self.__class__.__name__}/{self.objective.obj_name}/"
        os.makedirs(path, exist_ok=True)
        path = os.path.join(
            path, f"arc_{self.arc_string}_epochs_{self.num_epochs}_inner_{self.num_inner_iter}_lr_{self.learning_rate}_seed_{self.seed}.nnx"
        )
        checkpointer_model = ocp.StandardCheckpointer()

        if isinstance(self.policy, nnx.Module):
            _, state = nnx.split(self.policy)
            if os.path.exists(path): shutil.rmtree(path)
            checkpointer_model.save(path, state)
        elif isinstance(self.policy, list): # NOTE: for generalized (composite) models
            for i, submodel in enumerate(self.submodels):
                _, state = nnx.split(submodel.policy)
                subpath = path.replace(".nnx", f"_{i}.nnx")
                if os.path.exists(subpath): shutil.rmtree(subpath)
                checkpointer_model.save(subpath, state)
        else:
            raise ValueError(f"Policy type {type(self.policy)} not supported")
        print(f"Successfully saved policy ", end="")
        if dictionary is not None: # for losses, samples, etc
            checkpointer_misc = ocp.PyTreeCheckpointer()
            subpath = path.replace(".nnx", ".dict")
            if os.path.exists(subpath): shutil.rmtree(subpath)
            checkpointer_misc.save(subpath, dictionary)
            print(f"(and losses, samples, etc) ", end="")
        print(f"to {path}")

    def load(self):
        """Load the density model."""
        if self.policy is None:
            raise ValueError("Must call load() in precompute(), or after policy is initialized.")
        path = f"{os.environ['HOME']}/function-decomposed_EDA/checkpoints/policy/{self.__class__.__name__}/{self.objective.obj_name}/"
        path = os.path.join(
            path, f"arc_{self.arc_string}_epochs_{self.num_epochs}_inner_{self.num_inner_iter}_lr_{self.learning_rate}_seed_{self.seed}.nnx"
        )
        checkpointer_model = ocp.StandardCheckpointer()
        if isinstance(self.policy, nnx.Module):
            graphdef, state = nnx.split(self.policy)
            state = checkpointer_model.restore(path, state)
            self.policy = nnx.merge(self.policy, state)
        elif isinstance(self.policy, list): # NOTE: for generalized (composite) models
            for i, submodel in enumerate(self.submodels):
                graphdef, state = nnx.split(submodel)
                state = checkpointer_model.restore(path.replace(".nnx", f"_{i}.nnx"), state)
                self.submodels[i] = nnx.merge(graphdef, state)
        print(f"Successfully loaded policy from {path}")

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

    def get_mode(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the mode of the density model.
        """
        if self.policy is None:
            raise ValueError("Must call precompute() before get_mode()")
        
        self.policy.eval() # make sure dropout is off
        
        x, log_prob = self.policy.sample_mode() # If don't pass rng --> uses argmax
        return x, self.objective.query(x[None, ...]).squeeze()