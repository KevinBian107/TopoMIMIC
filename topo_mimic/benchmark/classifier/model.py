"""Flax MLP classifier adapted from track-mjx discriminator_network.py.

Changed from 1 binary logit to N=len(TOP_8_TRANSITIONS) logits.
Input: (B, K, D) flattened to (B, K*D).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


class BehaviorClassifier(nn.Module):
    n_classes: int
    hidden_sizes: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        for i, h in enumerate(self.hidden_sizes):
            x = nn.Dense(h, name=f"hidden_{i}")(x)
            x = nn.silu(x)
            x = nn.LayerNorm(name=f"ln_{i}")(x)
            x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        return nn.Dense(self.n_classes, name="logits")(x)


def init_params(
    model: BehaviorClassifier,
    rng: jax.Array,
    example_input: jnp.ndarray,
) -> dict:
    return model.init({"params": rng, "dropout": rng}, example_input, training=False)


def apply_eval(model: BehaviorClassifier, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Deterministic forward pass for eval/inference."""
    return model.apply(params, x, training=False)
