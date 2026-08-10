"""Memory-bounded deterministic sampling for very large robot datasets."""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any, Iterator, Sized

from torch.utils.data import Sampler


_MASK64 = (1 << 64) - 1


def _mix64(value: int) -> int:
    """SplitMix64 finalizer: deterministic counter-based pseudo-randomness."""

    value = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


class BudgetedResumableSampler(Sampler[tuple[int, int]]):
    """Sample a fixed virtual epoch without allocating ``randperm(N)``.

    Sampling is intentionally with replacement.  A fixed counter, seed and
    epoch make every position reproducible and permit O(1) resume skipping.
    ``episode_uniform`` first samples an episode uniformly, then a valid window
    within it, preventing long trajectories from dominating the objective.

    ``task_hierarchical`` consumes a compact, immutable manifest generated from
    BEHAVIOR-1K episode metadata and annotations.  It first samples a task with
    a bounded complexity weight, then an episode uniformly, and finally mixes
    natural-time, skill and skill-boundary windows.  The manifest is tiny
    compared with the 210M-frame corpus and the counter-based stream retains
    exact O(1) resume semantics.
    """

    STRATEGIES = {"frame_uniform", "episode_uniform", "task_hierarchical"}

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        *,
        strategy: str = "frame_uniform",
        samples_per_epoch: int | None = None,
        drop_padded_windows: bool = True,
        sampling_manifest_path: str | None = None,
        natural_probability: float = 0.70,
        skill_probability: float = 0.20,
        boundary_probability: float = 0.10,
        task_block_size: int = 1,
        task_reuse_steps: int = 1,
        episode_batch_locality: bool = False,
        episode_reuse_steps: int = 1,
        window_batch_locality_span: int = 0,
    ) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.strategy = str(strategy).strip()
        if self.strategy not in self.STRATEGIES:
            raise ValueError(
                f"sampling strategy must be one of {sorted(self.STRATEGIES)}, "
                f"got {self.strategy!r}"
            )
        dataset_size = len(dataset)
        if dataset_size <= 0:
            raise ValueError("cannot sample an empty dataset")
        self.samples_per_epoch = (
            dataset_size if samples_per_epoch is None else int(samples_per_epoch)
        )
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if self.batch_size <= 0 or self.num_processes <= 0:
            raise ValueError("batch_size and num_processes must be positive")
        self.drop_padded_windows = bool(drop_padded_windows)
        self.sampling_manifest_path = (
            str(sampling_manifest_path).strip() if sampling_manifest_path else ""
        )
        probabilities = (
            float(natural_probability),
            float(skill_probability),
            float(boundary_probability),
        )
        if any(value < 0.0 for value in probabilities) or abs(sum(probabilities) - 1.0) > 1e-8:
            raise ValueError(
                "natural/skill/boundary probabilities must be non-negative and sum to 1"
            )
        self._mode_cdf = (
            probabilities[0],
            probabilities[0] + probabilities[1],
            1.0,
        )
        self.task_block_size = int(task_block_size)
        if self.task_block_size <= 0:
            raise ValueError("task_block_size must be positive")
        self.task_reuse_steps = int(task_reuse_steps)
        if self.task_reuse_steps <= 0:
            raise ValueError("task_reuse_steps must be positive")
        self.episode_batch_locality = bool(episode_batch_locality)
        self.episode_reuse_steps = int(episode_reuse_steps)
        if self.episode_reuse_steps <= 0:
            raise ValueError("episode_reuse_steps must be positive")
        if self.episode_reuse_steps > 1 and not self.episode_batch_locality:
            raise ValueError(
                "episode_reuse_steps > 1 requires episode_batch_locality"
            )
        self.window_batch_locality_span = int(window_batch_locality_span)
        if self.window_batch_locality_span < 0:
            raise ValueError("window_batch_locality_span must be non-negative")
        if self.window_batch_locality_span > 0 and not self.episode_batch_locality:
            raise ValueError(
                "window_batch_locality_span > 0 requires episode_batch_locality"
            )
        self.epoch = 0
        self.resume_batch_offset = 0
        self._episode_bounds = (
            self._resolve_episode_bounds()
            if self.strategy in {"episode_uniform", "task_hierarchical"}
            else ()
        )
        self._hierarchy = (
            self._load_hierarchy() if self.strategy == "task_hierarchical" else None
        )

    def _resolve_episode_bounds(self) -> tuple[tuple[int, int], ...]:
        if getattr(self.dataset, "_motion_index", None) is not None:
            raise ValueError(
                "episode_uniform cannot be combined with motion_ranges_path"
            )
        base = getattr(self.dataset, "lerobot_dataset", self.dataset)
        episode_index = getattr(base, "episode_data_index", None)
        if not isinstance(episode_index, dict) or not {"from", "to"} <= set(
            episode_index
        ):
            raise ValueError(
                "episode_uniform requires dataset.lerobot_dataset.episode_data_index"
            )
        starts = [int(value) for value in episode_index["from"]]
        stops = [int(value) for value in episode_index["to"]]
        if len(starts) != len(stops):
            raise ValueError("episode_data_index from/to length mismatch")
        horizon = int(getattr(base, "obs_size", 1))
        bounds = []
        for start, stop in zip(starts, stops, strict=True):
            valid_stop = stop - max(horizon - 1, 0) if self.drop_padded_windows else stop
            if valid_stop > start:
                bounds.append((start, valid_stop))
        if not bounds:
            raise ValueError(
                "episode_uniform found no episode long enough for the configured horizon"
            )
        return tuple(bounds)

    def _load_hierarchy(self) -> dict[str, Any]:
        if not self.sampling_manifest_path:
            raise ValueError(
                "task_hierarchical requires sampling_manifest_path"
            )
        path = Path(self.sampling_manifest_path).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read sampling manifest {path}: {exc}") from exc
        if payload.get("schema_version") != "1.0":
            raise ValueError(
                f"unsupported sampling manifest schema: {payload.get('schema_version')!r}"
            )
        episodes = payload.get("episodes")
        tasks = payload.get("tasks")
        if not isinstance(episodes, list) or len(episodes) != len(self._episode_bounds):
            raise ValueError(
                "sampling manifest episodes must align exactly with the selected dataset "
                f"({len(episodes) if isinstance(episodes, list) else 'invalid'} != "
                f"{len(self._episode_bounds)})"
            )
        normalized_episodes: list[dict[str, Any]] = []
        horizon = int(payload.get("horizon", 1))
        for position, (entry, (episode_start, episode_stop)) in enumerate(
            zip(episodes, self._episode_bounds, strict=True)
        ):
            if not isinstance(entry, dict):
                raise ValueError(f"manifest episode {position} must be a mapping")
            raw_length = int(entry.get("length", -1))
            base = getattr(self.dataset, "lerobot_dataset", self.dataset)
            raw_starts = getattr(base, "episode_data_index")["from"]
            raw_stops = getattr(base, "episode_data_index")["to"]
            actual_length = int(raw_stops[position]) - int(raw_starts[position])
            if raw_length != actual_length:
                raise ValueError(
                    f"manifest episode length mismatch at position {position}: "
                    f"{raw_length} != {actual_length}"
                )
            valid_from = max(int(entry.get("valid_from", 0)), 0)
            valid_to = min(int(entry.get("valid_to", raw_length)), raw_length)
            # ``episode_stop`` is already the exclusive last valid window start
            # after accounting for the dataset observation horizon.
            valid_window_from = int(episode_start) + valid_from
            valid_window_to = min(
                int(episode_start) + max(valid_to - horizon + 1, valid_from),
                int(episode_stop),
            )
            if valid_window_to <= valid_window_from:
                raise ValueError(
                    f"manifest episode {entry.get('episode_index')} has no valid {horizon}-frame window"
                )
            segments = []
            for raw_segment in entry.get("skill_segments", []):
                if not isinstance(raw_segment, (list, tuple)) or len(raw_segment) != 2:
                    raise ValueError("skill_segments entries must be [start, stop]")
                start = max(int(raw_segment[0]), valid_from)
                stop = min(int(raw_segment[1]), valid_to)
                local_stop = min(int(episode_start) + stop, valid_window_to)
                local_start = min(max(int(episode_start) + start, valid_window_from), local_stop)
                if local_stop > local_start:
                    segments.append((local_start, local_stop))
            boundaries = []
            for raw_boundary in entry.get("boundaries", []):
                boundary = int(episode_start) + int(raw_boundary)
                if valid_window_from <= boundary < int(episode_start) + valid_to:
                    boundaries.append(boundary)
            normalized_episodes.append(
                {
                    "episode_index": int(entry["episode_index"]),
                    "task_index": int(entry["task_index"]),
                    "start": valid_window_from,
                    "stop": valid_window_to,
                    "segments": tuple(segments),
                    "boundaries": tuple(boundaries),
                    "horizon": horizon,
                }
            )
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("sampling manifest tasks must be a non-empty list")
        task_entries = []
        cumulative_weights = []
        total_weight = 0.0
        for task in tasks:
            positions = tuple(int(value) for value in task.get("episode_positions", []))
            if not positions or any(not 0 <= value < len(normalized_episodes) for value in positions):
                raise ValueError(f"task {task.get('task_index')} has invalid episode positions")
            task_index = int(task["task_index"])
            if any(normalized_episodes[value]["task_index"] != task_index for value in positions):
                raise ValueError(f"task {task_index} episode positions cross task boundaries")
            weight = float(task.get("weight", 1.0))
            if not 0.0 < weight < float("inf"):
                raise ValueError(f"task {task_index} has invalid weight {weight}")
            total_weight += weight
            cumulative_weights.append(total_weight)
            task_entries.append({"task_index": task_index, "positions": positions})
        return {
            "path": str(path),
            "episodes": tuple(normalized_episodes),
            "tasks": tuple(task_entries),
            "task_cdf": tuple(cumulative_weights),
            "task_weight_total": total_weight,
        }

    @staticmethod
    def _unit_interval(value: int) -> float:
        return (int(value) & _MASK64) / float(1 << 64)

    def _hierarchical_sample(self, counter: int, epoch_key: int) -> tuple[int, int]:
        assert self._hierarchy is not None
        micro_batch_index = int(counter) // self.batch_size
        # Keep a small configurable block on one task to reduce metadata/video
        # churn without forcing an entire global optimizer step onto one task.
        if self.task_reuse_steps > 1:
            # Accelerate shards consecutive micro-batches across ranks.  Keep
            # each logical rank lane on one task for a few micro-steps while
            # different lanes still cover many tasks concurrently.  This
            # improves per-node MP4/page-cache reuse without making an entire
            # global optimizer step single-task.
            rank_lane = micro_batch_index % self.num_processes
            micro_step = micro_batch_index // self.num_processes
            task_counter = (
                rank_lane
                + (micro_step // self.task_reuse_steps) * self.num_processes
            )
        else:
            task_counter = int(counter) // self.task_block_size
        task_random = _mix64(epoch_key ^ task_counter ^ 0xA0761D6478BD642F)
        target = self._unit_interval(task_random) * self._hierarchy["task_weight_total"]
        task_position = bisect.bisect_left(self._hierarchy["task_cdf"], target)
        task = self._hierarchy["tasks"][min(task_position, len(self._hierarchy["tasks"]) - 1)]

        if self.episode_batch_locality:
            # Accelerate assigns consecutive global micro-batches to rank
            # lanes.  Reuse within each lane, not across adjacent ranks:
            # lane r observes r, r+world, r+2*world, ... .
            episode_rank_lane = micro_batch_index % self.num_processes
            episode_micro_step = micro_batch_index // self.num_processes
            episode_counter = (
                episode_rank_lane
                + (episode_micro_step // self.episode_reuse_steps)
                * self.num_processes
            )
        else:
            episode_counter = int(counter)
        episode_random = _mix64(
            epoch_key ^ episode_counter ^ 0xE7037ED1A0B428DB
        )
        episode_position = task["positions"][episode_random % len(task["positions"])]
        episode = self._hierarchy["episodes"][episode_position]

        # Episode locality must not collapse all B windows to the same frame.
        # Keep the mode/window draw counter-specific after selecting one shared
        # episode for the microbatch.
        if self.window_batch_locality_span > 0:
            # One mode/region per global micro-batch, but retain a distinct
            # counter-specific window draw for every sample in B.
            mode_seed = _mix64(
                epoch_key ^ micro_batch_index ^ 0xD1B54A32D192ED03
            )
            position_random = _mix64(
                epoch_key ^ int(counter) ^ 0x589965CC75374CC3
            )
        else:
            mode_seed = (
                _mix64(epoch_key ^ int(counter) ^ 0xD1B54A32D192ED03)
                if self.episode_batch_locality
                else episode_random
            )
            position_random = None
        mode_random = _mix64(mode_seed ^ 0x8EBC6AF09C88C6E3)
        region_random = _mix64(mode_random ^ 0x589965CC75374CC3)
        if position_random is None:
            position_random = region_random
        mode_value = self._unit_interval(mode_random)
        if mode_value < self._mode_cdf[0]:
            start, stop = episode["start"], episode["stop"]
        elif mode_value < self._mode_cdf[1] and episode["segments"]:
            start, stop = episode["segments"][region_random % len(episode["segments"])]
        elif episode["boundaries"]:
            boundary = episode["boundaries"][region_random % len(episode["boundaries"])]
            radius = max(int(episode["horizon"]) - 1, 1)
            start = max(int(episode["start"]), boundary - radius)
            stop = min(int(episode["stop"]), boundary + 1)
        else:
            start, stop = episode["start"], episode["stop"]
        if stop <= start:
            start, stop = episode["start"], episode["stop"]
        locality_span = self.window_batch_locality_span
        if locality_span > 0 and int(stop) - int(start) > locality_span:
            local_start = int(start) + _mix64(
                region_random ^ 0x94D049BB133111EB
            ) % (int(stop) - int(start) - locality_span + 1)
            start, stop = local_start, local_start + locality_span
        frame_index = int(start) + _mix64(position_random ^ 0x1D8E4E27C47D124F) % (int(stop) - int(start))
        return frame_index, position_random

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int) -> None:
        # Trainer checkpoints store an absolute epoch, not an additive offset.
        self.epoch = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int) -> None:
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self) -> None:
        self.resume_batch_offset = 0

    def _sample(self, counter: int) -> tuple[int, int]:
        epoch_key = _mix64(self.seed ^ _mix64(self.epoch))
        first = _mix64(epoch_key ^ int(counter))
        second = _mix64(first ^ 0xD1B54A32D192ED03)
        if self.strategy == "frame_uniform":
            frame_index = first % len(self.dataset)
        elif self.strategy == "episode_uniform":
            episode_start, episode_stop = self._episode_bounds[
                first % len(self._episode_bounds)
            ]
            frame_index = episode_start + second % (episode_stop - episode_start)
        else:
            frame_index, second = self._hierarchical_sample(counter, epoch_key)
        # Keep image augmentation statistically independent from the sampled
        # episode/window position while retaining a stateless resume stream.
        augmentation_seed = _mix64(second ^ 0x94D049BB133111EB) & (
            (1 << 63) - 1
        )
        return int(frame_index), int(augmentation_seed)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        sample_offset = 0
        if self.resume_batch_offset > 0:
            sample_offset = (
                self.resume_batch_offset * self.batch_size * self.num_processes
            )
        return iter(
            self._sample(counter)
            for counter in range(sample_offset, self.samples_per_epoch)
        )

    def __len__(self) -> int:
        return max(
            self.samples_per_epoch
            - self.resume_batch_offset * self.batch_size * self.num_processes,
            0,
        )
