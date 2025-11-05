import jax
import jax.numpy as jnp
import optax as ox
from flax import nnx
from typing import Optional, Tuple, List
import os
import orbax.checkpoint as ocp

from src.problems.real.base import RealObjective
from src.models.regression.MLP import JointRegressionMLP


class OracleObjective(RealObjective):
    """Base class for oracle objective functions.
    This is a wrapper around a real objective function that allows 
    for training a model to approximate the objective function, so
    that it can be queried anywhere.
    Also used to access objective functions by name instead of 
    individually, even if not training a model.
    """
    
    def __init__(self, obj_name: str, *args, **kwargs):
        """
        Initialize a binary objective function.
        
        Args:
            D: Dimension of each position in the sequence
            L: Length of sequence
            maximize: Whether to maximize (True) or minimize (False) the objective
            best_x: Global optimum of the objective (if known)
        """
        self.obj_name = obj_name
        self.obj = self._get_obj(obj_name)
        self.WT_seq = self.obj.WT
        self.active_inds = self.obj.active_inds
        self.L = len(self.WT_seq)
        self.WT = self.obj.str_to_int(self.WT_seq).squeeze()[None, :]
        self.genotypes = self.obj.genotypes
        self.values = self.obj.values
        self.batch_size = self.obj.values.size # can be changed in `train`
        self.D = self.obj.D
        self.best_x = None
        self.fname = f"{os.getcwd()}/checkpoints/oracle/{obj_name}/"
    
    def _get_obj(self, obj_name: str) -> RealObjective:
        match obj_name.upper():
            case "GB1_55":
                from src.problems.real.nonbinary.gb1_55 import GB1_55
                self.obj_name_pretty = 'GB1'
                return GB1_55()
            case "TDP43":
                from src.problems.real.nonbinary.tdp43 import TDP43
                self.obj_name_pretty = 'TDP-43'
                return TDP43()
            case "GFP":
                from src.problems.real.nonbinary.gfp import GFP
                self.obj_name_pretty = 'GFP'
                return GFP()
            case "AMYLOID":
                from src.problems.real.nonbinary.amyloid import Amyloid
                self.obj_name_pretty = 'Amyloid beta'
                return Amyloid()
            case "AAV":
                from src.problems.real.nonbinary.aav import AAV
                self.obj_name_pretty = 'AAV2 capsid'
                return AAV()
            case "YNZC":
                from src.problems.real.nonbinary.tsuboyama.ynzc_bacsu import YNZC
                self.obj_name_pretty = 'ynzC'
                return YNZC()
            case "GCN4":
                from src.problems.real.nonbinary.gcn4 import GCN4
                self.obj_name_pretty = 'Gcn4'
                return GCN4()
            case "HIS7":
                from src.problems.real.nonbinary.his7 import HIS7
                self.obj_name_pretty = 'HIS7'
                return HIS7()
            case "PHOT":
                from src.problems.real.nonbinary.phot import PHOT
                self.obj_name_pretty = 'CreiLOV'
                return PHOT()
            case _:
                raise ValueError(f"Objective {obj_name} not found")
        
    def train(self, 
              num_epochs: int, 
              num_inner_iter: int,
              batch_size: int,
              hidden_dims: List[int] = [64, 128, 64], 
              learning_rate: float = 1e-3,
              seed: int = 0,
              save_load: bool = True,
              verbose: bool = False):
        self.model = JointRegressionMLP(
            n_states=self.obj.D,
            hidden_dims=hidden_dims,
            rngs=jax.random.PRNGKey(seed)
        )
        if batch_size > self.values.size:
            batch_size = self.values.size
        self.batch_size = batch_size

        if save_load:
            self.fname = os.path.join(
                self.fname, f"arc_{'-'.join(map(str, hidden_dims))}_epochs_{num_epochs}_inner_{num_inner_iter}_lr_{learning_rate}_batch_{batch_size}_seed_{seed}.nnx"
            )
            checkpointer_model = ocp.StandardCheckpointer()
            checkpointer_misc = ocp.PyTreeCheckpointer()
            if os.path.exists(self.fname):
                graphdef, state = nnx.split(self.model)
                state = checkpointer_model.restore(self.fname, state)
                self.model = nnx.merge(graphdef, state)
                self.model.eval()
                loaded_data = checkpointer_misc.restore(self.fname.replace(".nnx", ".dict"))
                losses = loaded_data["losses"]
                print(f"Successfully loaded model from {self.fname}")
                return losses
        optimizer = ox.adamw(learning_rate)
        nnx_optimizer = nnx.Optimizer(self.model, optimizer)

        def loss_fn(model, x, y):
            return ((model.forward(x) - y)**2).mean()
        
        @nnx.jit
        def update_step(model, optimizer, x, y):
            loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
            optimizer.update(grads)
            return model, optimizer, loss
        
        def inner_step(carry, _):
            model, optimizer, x, y = carry
            model, optimizer, loss = update_step(model, optimizer, x, y)
            return (model, optimizer, x, y), loss
        
        scan_inner = nnx.scan(
            f=inner_step, length=num_inner_iter
        )
        
        def epoch_step(carry, _):
            model, optimizer, rng = carry
            rng, rng_shuffle = jax.random.split(rng)
            batch_inds = jax.random.choice(
                rng_shuffle, len(self.values), (batch_size,), replace=False
            )
            inner_carry = (
                model, optimizer, 
                self.genotypes[batch_inds, :], 
                self.values[batch_inds]
            )
            (model, optimizer, _, _), losses = scan_inner(
                inner_carry, jnp.arange(num_inner_iter)
            )
            
            return (model, optimizer, rng), losses[-1]
        
        scan_epoch = nnx.scan(
            f=epoch_step, length=num_epochs
        )

        self.model.train()
        init_carry = (self.model, nnx_optimizer, jax.random.PRNGKey(seed+1))
        (self.model, _, _), losses = scan_epoch(
            init_carry, jnp.arange(num_epochs)
        )
        self.model.eval()
        if save_load:
            graphdef, state = nnx.split(self.model)
            try:
                checkpointer_model.save(self.fname, state)
                checkpointer_misc.save(self.fname.replace(".nnx", ".dict"), {"losses": losses})
                print(f"Successfully saved model (and losses) to {self.fname}")
            except Exception as e:
                # NOTE: oracles are common, so another process may have saved in meantime. 
                state = checkpointer_model.restore(self.fname, state)
                self.model = nnx.merge(graphdef, state)
                self.model.eval()
                loaded_data = checkpointer_misc.restore(self.fname.replace(".nnx", ".dict"))
                losses = loaded_data["losses"]
                print(f"Training intercepted. Threw out and successfully loaded model from {self.fname}.")
        return losses
        
    def query(self, x: jnp.ndarray, batch_size: Optional[int] = None) -> float:
        if batch_size is None: batch_size = self.batch_size
        res = []
        for i in range(0, len(x), batch_size):
            res.append(self.model.forward(x[i:i+batch_size, :]))
        return jnp.concatenate(res)

    def query_nobatch(self, x: jnp.ndarray) -> float:
        return self.model.forward(x)
    
    def query_all(self, batch_size: Optional[int] = None) -> jnp.ndarray:
        return self.query(self.genotypes, batch_size)
    
    def query_WT(self) -> float:
        return self.model.forward(self.WT).squeeze()

    def query_nobatch(self, x):
        return self.model.forward(x)
