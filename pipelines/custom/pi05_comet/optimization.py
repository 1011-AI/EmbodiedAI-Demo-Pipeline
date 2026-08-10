"""Long-horizon schedules used by Demo Pipeline's Comet continuation."""

from __future__ import annotations

import dataclasses

import optax


@dataclasses.dataclass(frozen=True)
class WarmupStableDecaySchedule:
    """WSD schedule with an optional final decay stage."""

    peak_lr: float = 2.5e-6
    warmup_steps: int = 1_000
    stable_steps: int = 99_000
    decay_steps: int = 0
    end_lr: float = 2.5e-7

    def create(self) -> optax.Schedule:
        if self.peak_lr <= 0 or self.warmup_steps < 0 or self.stable_steps < 0:
            raise ValueError("invalid WSD schedule")
        schedules: list[optax.Schedule] = []
        boundaries: list[int] = []
        if self.warmup_steps:
            schedules.append(
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                )
            )
            boundaries.append(self.warmup_steps)
        schedules.append(optax.constant_schedule(self.peak_lr))
        if self.decay_steps:
            boundaries.append(self.warmup_steps + self.stable_steps)
            alpha = self.end_lr / self.peak_lr
            schedules.append(
                optax.cosine_decay_schedule(
                    init_value=self.peak_lr,
                    decay_steps=self.decay_steps,
                    alpha=alpha,
                )
            )
        if len(schedules) == 1:
            return schedules[0]
        return optax.join_schedules(schedules, boundaries)
