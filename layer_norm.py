"""
Reimplemented this from
https://github.com/google/flax/blob/b5f478fff4d9dd803c61efe16db983c8a62817c0/flax/nnx/nn/normalization.py#L382

"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union, Iterable
from jax import lax
import flax.linen as nn
import jax
import jax.numpy as jnp
import flax.linen as nn


class LayerNorm(nn.Module):
    features: int
    epsilon: float = 1e-6
    use_bias: bool = True
    use_scale: bool = True
    reduction_axes: int = -1
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Compute mean and variance
        reduction_axes = tuple(
            i if i >= 0 else x.ndim + i for i in (self.reduction_axes,)
        )
        mean = jnp.mean(x, axis=reduction_axes, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=reduction_axes, keepdims=True)

        # Normalize
        centered = x - mean
        normed = centered * jax.lax.rsqrt(var + self.epsilon)

        # Optionally apply scale and bias
        if self.use_scale:
            scale = self.param("scale", nn.initializers.ones, (self.features,))
            normed = normed * scale.reshape((1,) * (x.ndim - 1) + (-1,))
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,))
            normed = normed + bias.reshape((1,) * (x.ndim - 1) + (-1,))
        return normed, centered
