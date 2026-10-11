from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["weights"],
    meta_fields=["in_features", "out_features"],
)
@dataclass(frozen=True)
class FourierFeatures1d:
    weights: jax.Array
    in_features: int
    out_features: int


def init(key: Any, in_features: int, out_features: int, dtype=jnp.float32) -> FourierFeatures1d:

    if in_features < 1 or out_features < 1:
        raise ValueError()

    weights = jax.random.uniform(
        key,
        (out_features, in_features),
        dtype=dtype,
        minval=-np.sqrt(1 / in_features),
        maxval=np.sqrt(1 / in_features),
    )

    return FourierFeatures1d(weights, in_features, out_features)


def forward(params: FourierFeatures1d, x: jax.Array) -> jax.Array:

    freqs = 2 * jnp.pi * np.arange(params.out_features, dtype=params.weights.dtype)

    x = freqs * jnp.einsum("ij,...j->...i", params.weights, x)

    return jnp.concat((jnp.sin(x), jnp.cos(x)), axis=-1)
