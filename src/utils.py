import jax
import jax.numpy as jnp
from flax import nnx
import math

def print_nnx_param_shapes(module: nnx.Module):
    _, state = nnx.split(module)
    
    def _print_shapes(d, prefix=""):
        if isinstance(d, dict) or isinstance(d, nnx.State):
            for k, v in d.items():
                _print_shapes(v, prefix + f"{k}/")
        elif hasattr(d, "value") and isinstance(d.value, jax.Array):
            print(f"{prefix[:-1]}: shape = {d.value.shape}")
    
    _print_shapes(state)

def count_params(module: nnx.Module):
    _, state = nnx.split(module)

    def _count_params(d, prefix=""):
        count = 0
        if isinstance(d, dict) or isinstance(d, nnx.State):
            for k, v in d.items():
                count += _count_params(v, prefix + f"{k}/")
        elif hasattr(d, "value") and isinstance(d.value, jax.Array):
            count += d.value.size
        return count
    
    return int(_count_params(state))

def find_nan_params(module: nnx.Module):
    _, state = nnx.split(module)
    
    def _find_nans(d, prefix=""):
        if isinstance(d, dict) or isinstance(d, nnx.State):
            for k, v in d.items():
                _find_nans(v, prefix + f"{k}/")
        elif hasattr(d, "value") and isinstance(d.value, jax.Array):
            if jnp.isnan(d.value).any():
                param_name = prefix[:-1]  # Remove trailing slash
                nan_count = jnp.isnan(d.value).sum()
                inf_count = jnp.isinf(d.value).sum()
                if nan_count > 0 or inf_count > 0:
                    print(f"{param_name} {d.value.shape}")
                    if nan_count > 0:  print(f"\tnan count: {nan_count} / {d.value.size}")
                    if inf_count > 0: print(f"\tinf count: {inf_count} / {d.value.size}")
                else:
                    print(f"{param_name} {d.value.shape}")
                    print(f"\tclean!")
    
    _find_nans(state)


def fix_nbatch_smallest_batch_size(batch_size_graph, n_indices):
    """
    Fix nbatches to be ceiling(n_indices / batch_size_graph).
    Then, find the smallest batch size that still gets ≤ nbatches.
    This should alleviate needless memory strain while still taking the same
    number of batch iterations.
    """
    max_batches = math.ceil(n_indices / batch_size_graph)
    # smallest batch size that still gets ≤ max_batches
    return math.ceil(n_indices / max_batches), max_batches


def jaxrng_to_int(rng: jax.random.PRNGKey) -> int:
    """
    Convert a JAX PRNGKey to an integer seed.
    """
    return jax.random.randint(rng, (1,), 0, jnp.iinfo(jnp.int32).max).item()


def kl_divergence_uniform_means(uniform_lp, model_lp):
    return uniform_lp - model_lp # both expected to be log probs, and length-normalized.

def kl_divergence(samples, p, q, batch_size=2**10):
    """
    Compute KL divergence between two distributions:
        E_p[log p(x) - log q(x)]
    Assumes that samples are drawn from p.
    """
    p_lps, q_lps = jnp.zeros(samples.shape[0]), jnp.zeros(samples.shape[0])
    for i in range(0, samples.shape[0], batch_size): # TODO: scan / jit?
        batch = samples[i:min(i+batch_size, samples.shape[0])]
        # NOTE: both expected to be log probs, and length-normalized.
        p_lps = p_lps.at[i:min(i+batch_size, samples.shape[0])].set(p.likelihood(batch))
        q_lps = q_lps.at[i:min(i+batch_size, samples.shape[0])].set(q.likelihood(batch))
    return p_lps.mean() - q_lps.mean()

def bottom_percentile_indices(arr, p=0.5):
    """
    Return the indices of elements in arr that are in the bottom p percentile.

    Parameters:
        arr (jax.numpy.ndarray): Input array.
        p (float): Quantile threshold (0-1).

    Returns:
        jax.numpy.ndarray: Indices of elements below the p-th percentile.
    """
    threshold = jnp.quantile(arr.ravel(), p)
    return jnp.where(arr <= threshold)[0]