from __future__ import annotations

import jax
import jax.numpy as jnp

from openpi.models import gemma
from openpi.models.pi0_config import Pi0Config


def test_xla_attention_matches_released_einsum_path() -> None:
    config = gemma.get_config("dummy")
    xs = [
        jax.random.normal(
            jax.random.key(1), (2, 5, config.width), dtype=jnp.bfloat16
        ),
        jax.random.normal(
            jax.random.key(2), (2, 3, config.width), dtype=jnp.bfloat16
        ),
    ]
    positions = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (2, 8))
    mask = jnp.ones((2, 1, 8, 8), dtype=jnp.bool_)
    released = gemma.Attention(
        configs=(config, config), implementation="einsum"
    )
    variables = released.init(jax.random.key(3), xs, positions, mask, None)
    expected, _ = released.apply(variables, xs, positions, mask, None)
    actual, _ = gemma.Attention(
        configs=(config, config), implementation="xla"
    ).apply(variables, xs, positions, mask, None)
    for reference, candidate in zip(expected, actual, strict=True):
        assert jnp.array_equal(reference, candidate)


def test_pi05_graph_performance_policies_are_parameter_shape_neutral() -> None:
    baseline = Pi0Config(pi05=True, action_horizon=32, max_token_len=256)
    optimized = Pi0Config(
        pi05=True,
        action_horizon=32,
        max_token_len=256,
        gemma_remat_policy="dots_with_no_batch_dims_saveable",
        siglip_remat_policy="dots_with_no_batch_dims_saveable",
        attention_implementation="cudnn",
    )
    assert baseline.inputs_spec() == optimized.inputs_spec()
    assert baseline.action_dim == optimized.action_dim == 32
