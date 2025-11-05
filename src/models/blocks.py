# mlp blocks, transformer blocks, etc.
import math 

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Sequence, List


class Linear(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.weight = nnx.Param(nnx.initializers.normal(0.02)(rngs.param(), (in_dim, out_dim)))
        self.bias = nnx.Param(jnp.zeros((out_dim,)))
    
    def __call__(self, x: jax.Array):
        return x @ self.weight + self.bias

class ZeroedLinear(nnx.Module): # 0 weights too...may be dangerous.
    def __init__(self, in_dim: int, out_dim: int):
        self.weight = nnx.Param(jnp.zeros((in_dim, out_dim)))
        self.bias = nnx.Param(jnp.zeros((out_dim,)))
    
    def __call__(self, x: jax.Array):
        return x @ self.weight + self.bias

class MLP(nnx.Module):
    def __init__(self, dims: list[int], *, rngs: nnx.Rngs, scaled: bool = False, nonneg: bool = False):
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i+1], rngs=rngs))
            if i < len(dims) - 2:
                self.layers.append(nnx.gelu)
                self.layers.append(nnx.Dropout(rate=0.1, rngs=rngs))
                self.layers.append(nnx.LayerNorm(dims[i+1], rngs=rngs))
        if scaled:
            self.layers.append(nnx.sigmoid) # NOTE: output is in [0, 1]
        elif nonneg:
            self.layers.append(nnx.relu) # NOTE: output is in [0, inf]

    def __call__(self, x: jax.Array):
        for layer in self.layers:
            x = layer(x)
        return x

class ZeroedMLP(nnx.Module):
    def __init__(self, dims: list[int], *, rngs: nnx.Rngs, scaled: bool = False, nonneg: bool = False):
        self.layers = []
        for i in range(len(dims) - 2):
            self.layers.append(Linear(dims[i], dims[i+1], rngs=rngs))
            self.layers.append(nnx.gelu)
            self.layers.append(nnx.Dropout(rate=0.1, rngs=rngs))
            self.layers.append(nnx.LayerNorm(dims[i+1], rngs=rngs))
        self.layers.append(ZeroedLinear(dims[-2], dims[-1]))
        if scaled:
            self.layers.append(nnx.sigmoid) # NOTE: output is in [0, 1]...w/ Zeroed, should give .5?
        elif nonneg: # sigmoid already nonneg, don't do again.
            self.layers.append(nnx.relu) # NOTE: output is in [0, inf]

    def __call__(self, x: jax.Array):
        for layer in self.layers:
            x = layer(x)
        return x

########## TRANSFORMERS ##########

class TransformerBlock(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, *, rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=d_model,
            qkv_features=d_model,
            out_features=d_model,
            dropout_rate=0.0,
            decode=False, # NOTE: only for sample-time
            rngs=rngs,
            dtype=jnp.bfloat16,
            # normalize_qk=True,
        )
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ff = nnx.Sequential(
            nnx.Linear(d_model, d_ff, rngs=rngs),
            nnx.gelu,
            nnx.Linear(d_ff, d_model, rngs=rngs)
        )
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(self, q, k, mask=None, decode=False):
        attn_out = self.attention(inputs_q=q, inputs_k=k, mask=mask, decode=decode)
        x = self.norm1(q + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out) # https://github.com/znowu/cliqueformer-code/blob/main/architectures/blocks.py
