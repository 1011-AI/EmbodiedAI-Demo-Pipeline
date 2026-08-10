"""Audited Comet train/eval steps and conservative image augmentation."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import augmax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.utils as training_utils


def conservative_preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: _model.Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = _model.IMAGE_KEYS,
    image_resolution: tuple[int, int] = _model.IMAGE_RESOLUTION,
) -> _model.Observation:
    """Use shared per-window color jitter and no geometric augmentation."""

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images missing {image_keys}: {list(observation.images)}")
    batch_shape = observation.state.shape[:-1]
    sub_rngs = None if rng is None else jax.random.split(rng, observation.state.shape[0])
    jitter = augmax.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            image = _model.image_tools.resize_with_pad(image, *image_resolution)
        if train:
            image = image / 2.0 + 0.5
            image = jax.vmap(jitter)(sub_rngs, image)
            image = image * 2.0 - 1.0
        images[key] = image
    masks = {
        key: (
            jnp.ones(batch_shape, dtype=jnp.bool_)
            if key not in observation.image_masks
            else jnp.asarray(observation.image_masks[key])
        )
        for key in images
    }
    return _model.Observation(
        images=images,
        image_masks=masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        pcd_xyz=observation.pcd_xyz,
    )


def install_conservative_augmentation() -> None:
    _model.preprocess_observation = conservative_preprocess_observation


def _group_norms(grads: nnx.State) -> dict[str, at.Array]:
    vision = nnx_utils.PathRegex(".*img.*")
    action_expert = nnx_utils.PathRegex(".*llm.*_1.*")
    all_llm = nnx_utils.PathRegex(".*llm.*")
    vlm = nnx.All(all_llm, nnx.Not(action_expert))
    other = nnx.All(nnx.Not(vision), nnx.Not(all_llm))
    return {
        "grad_norm/vision": optax.global_norm(grads.filter(vision)),
        "grad_norm/vlm": optax.global_norm(grads.filter(vlm)),
        "grad_norm/action_expert": optax.global_norm(grads.filter(action_expert)),
        "grad_norm/other": optax.global_norm(grads.filter(other)),
    }


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    *,
    gradient_audit_interval: int = 1,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    if gradient_audit_interval <= 0:
        raise ValueError("gradient_audit_interval must be positive")
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, step_rng, observation, actions):
        return jnp.mean(model.compute_loss(step_rng, observation, actions, train=True))

    observation, actions = batch
    step_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model,
        step_rng,
        observation,
        actions,
    )
    params = state.params.filter(config.trainable_filter)
    updates, opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=nnx.state(model),
        opt_state=opt_state,
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_state.params,
            ),
        )
    gradient_audit_due = jnp.logical_or(
        state.step == 0,
        state.step % gradient_audit_interval == 0,
    )
    gradient_metrics = jax.lax.cond(
        gradient_audit_due,
        lambda: {
            "grad_norm": optax.global_norm(grads),
            **_group_norms(grads),
        },
        lambda: {
            "grad_norm": jnp.asarray(0.0, dtype=jnp.float32),
            "grad_norm/vision": jnp.asarray(0.0, dtype=jnp.float32),
            "grad_norm/vlm": jnp.asarray(0.0, dtype=jnp.float32),
            "grad_norm/action_expert": jnp.asarray(0.0, dtype=jnp.float32),
            "grad_norm/other": jnp.asarray(0.0, dtype=jnp.float32),
        },
    )
    info = {
        "loss": loss,
        "lr": config.lr_schedule.create()(state.step),
        "gradient_audit_performed": gradient_audit_due.astype(jnp.float32),
        **gradient_metrics,
    }
    return new_state, info


@at.typecheck
def eval_step(
    state: training_utils.TrainState,
    rng: at.KeyArrayLike,
    batch: tuple[_model.Observation, _model.Actions],
) -> at.Array:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    return jnp.mean(model.compute_loss(rng, observation, actions, train=False))
