import optax as ox
from optax.contrib import muon
import jax
import jax.numpy as jnp
from flax import nnx
from time import time
import os
import shutil
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

from src.utils import kl_divergence_uniform_means

from src.opt.base import Optimizer

PRIOR_OPTIONS = ['none', 'init_only', 'weight_only', 'kld_only', 'init_weight', 'init_kld']


class ModelBasedOptimizer(Optimizer):
    """Base class for model-based optimization methods."""
    
    def __init__(self, 
                 *args, 
                 learning_rate: float = 1e-3,
                 use_prior: str = 'none',
                 beta_prior: float = 1.0,
                 prior_data: jnp.ndarray = None,
                 beta_entropy: float = 0.0, # not currently used
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.model = None
        self.opt_state = None
        assert beta_prior >= 0.0, "Beta prior must be non-negative."
        self.beta_prior = beta_prior
        assert use_prior in PRIOR_OPTIONS, f"Invalid prior option: {use_prior}. Must be one of {PRIOR_OPTIONS}."
        self.use_prior = use_prior
        self.prior_data = prior_data
        self.prior = None
        assert beta_entropy >= 0.0, "Beta entropy must be non-negative."
        self.beta_entropy = beta_entropy

    def fit_uniform_prior(self, n_epochs=1000, batch_size=2**9, lr=2e-3, verbose=False):
        '''
        Fit a prior to uniform distribution on samples (e.g., support of offline dataset). 
        The hyperparams used seem to work well for MLPs and transformers alike, across different datasets.
        Interestingly, muon works alright for MLPs on this, even though it doesn't look good for same models
        trained as EDAs.
        Was using 1000 epochs, 2**10 batch size, 2e-3 lr for distr. shift experiments.
        '''
        N = self.prior_data.shape[0]
        checkpoint_dir = f"{os.getcwd()}/checkpoints/prior/{self.__class__.__name__}/{self.objective.obj_name}/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # First try exact seed match
        exact_path = os.path.join(
            # checkpoint_dir, f"arc_{self.arc_string}_N{N}bWT_epochs_{n_epochs}_lr_{lr}_bs_{batch_size}_seed_{self.seed}.nnx"
            checkpoint_dir, f"t4.5_arc_{self.arc_string}_N{N}worst_epochs_{n_epochs}_lr_{lr}_bs_{batch_size}_seed_{self.seed}.nnx"
        )
        
        # TODO: add option to save, load so don't keep fitting this every time. 
        assert self.policy is not None, "Policy must be initialized before fitting prior. Assuming have same form."
        graphdef, state = nnx.split(self.policy)
        self.prior = nnx.merge(graphdef, state)
        
        if os.path.exists(exact_path):
            try:
                self.load_prior(exact_path, verbose=verbose)
                return
            except Exception as e:
                if verbose: print(f"Error loading exact seed prior from {exact_path}: {e}")
        
        # If exact seed not found, look for any matching prior with different seed
        import glob
        # Use exact_path as template, replace 'seed_{number}.nnx' with 'seed_*.nnx'
        import re
        pattern = re.sub(r'seed_\d+\.nnx$', r'seed_*.nnx', exact_path)
        matching_files = glob.glob(pattern)
        
        if matching_files:
            # Use the first available prior with any seed
            available_path = matching_files[0]
            # Extract seed from filename for warning
            import re
            seed_match = re.search(r'seed_(\d+)\.nnx$', available_path)
            available_seed = seed_match.group(1) if seed_match else "unknown"
            
            try:
                if verbose: print(f"Using prior with different seed ({available_seed} vs {self.seed})")
                self.load_prior(available_path, verbose=verbose)
                return
            except Exception as e:
                if verbose: print(f"Error loading available prior from {available_path}: {e}")
        
        print(f"No compatible prior found. Fitting new prior with seed {self.seed}.")
        uniform_lp = jnp.log(1.0 / N) / len(self.n_states) # L-normalized.
        uniform_ppl = jnp.exp(-uniform_lp)

        self.prior.train()
        nnx_optimizer = nnx.Optimizer(
            self.prior,
            muon(
                ox.warmup_cosine_decay_schedule(
                    init_value   = 0.0,            # LR starts at 0
                    peak_value   = lr,        # value reached after warm-up
                    warmup_steps = n_epochs // 10,   # linear warm-up length
                    decay_steps  = n_epochs,    # schedule continues to TOTAL_STEPS
                    end_value    = 0.0             # LR decays back to 0
                )
            )
        )
        # Define loss function (negative log likelihood)
        def loss_fn(model, samples):
            log_likelihoods = model.likelihood(samples)
            return -log_likelihoods.mean()  # Negative for gradient ascent
        
        @nnx.jit
        def update_step(model, optimizer, samples):
            with jax.default_matmul_precision('bfloat16'):
                loss, grads = nnx.value_and_grad(loss_fn)(model, samples)
            optimizer.update(grads)
            return model, optimizer, loss
        
        def epoch_step(carry, _):
            rkey, optimizer, model = carry

            # Sample batch from "true" distribution
            rkey, sample_key = jax.random.split(rkey)
            batch_samples = self.prior_data[jax.random.randint(sample_key, (batch_size,), 0, self.prior_data.shape[0])]

            model, optimizer, loss = update_step(model, optimizer, batch_samples)
            
            return (rkey, optimizer, model), loss
        
        scan_epoch = nnx.scan(
            f=epoch_step, length=n_epochs
        )
        print(f"\nFitting prior...")
        start_time = time()
        (self.rng, nnx_optimizer, self.prior), losses = scan_epoch(
            (self.rng, nnx_optimizer, self.prior), jnp.arange(n_epochs)
        )
        print(f"  Training completed in {time() - start_time:.2f}s")
        # for final loss, KL, PPL -- use all samples so that it's exact.
        model_lps = jnp.zeros(N)
        for i in range(0, N, batch_size):
            batch_samples = self.prior_data[i:min(i+batch_size, N)]
            model_lps = model_lps.at[i:min(i+batch_size, N)].set(self.prior.likelihood(batch_samples))
        model_lp = model_lps.mean()
        print(f"\t Final loss (full data): {-model_lp:.2f} vs. uniform {-uniform_lp:.2f}")
        print(f"\t Final PPL (full data): {jnp.exp(-model_lp):.2f} vs. uniform {uniform_ppl:.2f}") # length-normalized.
        kl_div = uniform_lp - model_lp
        print(f"\t Final KL (full data): {kl_div:.2f} vs. uniform {0.0:.2f}")
        # sever prior from policy params
        graphdef, state = nnx.split(self.prior)
        self.prior = nnx.merge(graphdef, state)
        self.prior.eval()
        self.save_prior(exact_path, {
            'losses': losses,
            'model_lp': model_lp,
            'uniform_lp': uniform_lp,
            'kl_div': kl_div,
            'ppl': jnp.exp(-model_lp),
            'uniform_ppl': uniform_ppl,
        })
        plt.plot(losses); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Prior Training Loss')
        plt.savefig(exact_path.replace(".nnx", "_loss.png"))
        print(f"Saved loss plot to {exact_path.replace('.nnx', '_loss.png')}")
        plt.close()

    def save_prior(self, path, dictionary: dict = None):
        """Save the density model."""
        if self.prior is None:
            raise ValueError("Must call fit_uniform_prior() before save_prior()")
        checkpointer_model = ocp.StandardCheckpointer()

        if isinstance(self.prior, nnx.Module):
            _, state = nnx.split(self.prior)
            if os.path.exists(path): shutil.rmtree(path)
            checkpointer_model.save(path, state)
        elif isinstance(self.prior, list): # NOTE: for generalized (composite) models
            raise ValueError("Generalized prior not supported yet.")
            for i, submodel in enumerate(self.submodels):
                _, state = nnx.split(submodel.policy)
                subpath = path.replace(".nnx", f"_{i}.nnx")
                if os.path.exists(subpath): shutil.rmtree(subpath)
                checkpointer_model.save(subpath, state)
        else:
            raise ValueError(f"Prior type {type(self.prior)} not supported")
        print(f"Successfully saved prior ", end="")
        if dictionary is not None: # for losses, samples, etc
            checkpointer_misc = ocp.PyTreeCheckpointer()
            subpath = path.replace(".nnx", ".dict")
            if os.path.exists(subpath): shutil.rmtree(subpath)
            checkpointer_misc.save(subpath, dictionary)
            print(f"(and losses, samples, etc) ", end="")
        print(f"to {path}")

    def load_prior(self, path, verbose=False):
        """Load the density model."""
        if self.prior is None:
            raise ValueError("Must call fit_uniform_prior() before load_prior()")
        checkpointer_model = ocp.StandardCheckpointer()
        if isinstance(self.prior, nnx.Module):
            graphdef, state = nnx.split(self.prior)
            state = checkpointer_model.restore(path, state)
            self.prior = nnx.merge(graphdef, state)
        elif isinstance(self.prior, list): # NOTE: for generalized (composite) models
            raise ValueError("Generalized prior not supported yet.")
            for i, submodel in enumerate(self.submodels):
                graphdef, state = nnx.split(submodel)
                state = checkpointer_model.restore(path.replace(".nnx", f"_{i}.nnx"), state)
                self.submodels[i] = nnx.merge(graphdef, state)
        else:
            raise ValueError(f"Prior type {type(self.prior)} not supported")
        self.prior.eval()
        if verbose: print(f"Successfully loaded prior from {path}")
        subpath = path.replace(".nnx", ".dict")
        if os.path.exists(subpath):
            dictionary = ocp.PyTreeCheckpointer().restore(subpath)
            if verbose:
                print(f"\t Final loss (full data): {-dictionary['model_lp']:.2f} vs. uniform {-dictionary['uniform_lp']:.2f}")
                print(f"\t Final PPL (full data): {jnp.exp(-dictionary['model_lp']):.2f} vs. uniform {dictionary['uniform_ppl']:.2f}") # length-normalized.
                print(f"\t Final KL (full data): {dictionary['kl_div']:.2f} vs. uniform {0.0:.2f}")
