"""Standalone FastWAM Hydra transform for BEHAVIOR-1K R1Pro state."""

from __future__ import annotations

from typing import Any


class R1ProPolicyStateTransform:
    """Project the official R1Pro 61-D state into FastWAM's 23-D proprio."""

    def __init__(self, key: str = "default") -> None:
        self.key = str(key)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        import torch

        try:
            raw = batch["state"][self.key]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"FastWAM batch must contain batch['state'][{self.key!r}]"
            ) from exc
        if not isinstance(raw, torch.Tensor):
            raise TypeError(f"R1Pro state must be a torch.Tensor, got {type(raw).__name__}")
        if raw.ndim < 1 or raw.shape[-1] != 61:
            raise ValueError(
                f"R1Pro observation.state must have shape [..., 61], got {tuple(raw.shape)}"
            )
        batch["state"][self.key] = torch.cat(
            (
                raw[..., 0:3],
                raw[..., 53:57],
                raw[..., 3:10],
                raw[..., 24:26].sum(dim=-1, keepdim=True),
                raw[..., 28:35],
                raw[..., 49:51].sum(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        return batch

    def backward(self, batch: dict[str, Any]) -> dict[str, Any]:
        # The state projection is not invertible.  FastWAM only needs action
        # denormalization on this path, so the canonical 23-D state stays intact.
        return batch
