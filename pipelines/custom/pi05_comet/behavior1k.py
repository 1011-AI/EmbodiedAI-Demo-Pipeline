"""BEHAVIOR-1K v3 data contract for OpenPI-Comet continuation.

The released Comet loader targets the 2025 dataset and advances a mutable GOP
stream from ``__getitem__``.  This adapter keeps the raw 2026 dataset read-only,
uses LeRobot v3 map-style indices, and connects Demo Pipeline's stateless
hierarchical sampler so a global optimizer step uniquely determines the next
window on every JAX process.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import hashlib
import json
import logging
import multiprocessing
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch
from torch.utils.data import Sampler

from embodied_demo.behavior1k.r1pro import RGB_VIDEO_KEYS
from pipelines.custom.fastwam.behavior1k.budget_sampler import (
    BudgetedResumableSampler,
)

import openpi.models.model as _model
import openpi.models.pi0_config as _pi0_config
import openpi.policies.b1k_policy as _b1k_policy
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


COMET_ASSET_ID = "behavior-1k/2025-challenge-demos"
COMET_DATASET_REPO_ID = "behavior-1k/2026-challenge-demos"
ACTION_HORIZON = 32
MAX_TOKEN_LEN = 256
RAW_STATE_DIM = 61
ACTION_DIM = 23
# 30 FPS timestamps stored as float32 can differ from MP4 PTS by just over
# 1e-4 seconds.  This matches the project's already-audited Behavior1K reader.
VIDEO_TOLERANCE_S = 5e-4


class Pi05CometDataError(RuntimeError):
    """Raised when data would violate the released Comet checkpoint contract."""


@dataclasses.dataclass(frozen=True)
class Behavior2026CometDataConfig(_config.DataConfigFactory):
    """Create Comet transforms while reading the 2026 LeRobot-v3 schema."""

    repo_id: str = COMET_DATASET_REPO_ID

    def create(
        self,
        assets_dirs: Path,
        model_config: _model.BaseModelConfig,
    ) -> _config.DataConfig:
        if model_config.model_type is not _model.ModelType.PI05:
            raise Pi05CometDataError("Behavior2026CometDataConfig requires PI0.5")
        if model_config.action_horizon != ACTION_HORIZON:
            raise Pi05CometDataError(
                f"Comet checkpoint requires action_horizon={ACTION_HORIZON}"
            )
        if model_config.max_token_len != MAX_TOKEN_LEN:
            raise Pi05CometDataError(
                "Behavior1K all-task language contract requires "
                f"max_token_len={MAX_TOKEN_LEN}, got {model_config.max_token_len}"
            )

        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/egocentric_camera": RGB_VIDEO_KEYS[0],
                        "observation/wrist_image_left": RGB_VIDEO_KEYS[1],
                        "observation/wrist_image_right": RGB_VIDEO_KEYS[2],
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                _b1k_policy.B1kInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[_b1k_policy.B1kOutputs(action_dim=ACTION_DIM)],
        )
        model_transforms = _config.ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            prompt_from_task=True,
            use_quantile_norm=True,
        )


def make_data_factory(
    *,
    dataset_root: str | Path,
    checkpoint_root: str | Path,
    episode_indices: Sequence[int],
) -> Behavior2026CometDataConfig:
    checkpoint = Path(checkpoint_root).expanduser().resolve()
    norm_file = checkpoint / "assets" / COMET_ASSET_ID / "norm_stats.json"
    if not norm_file.is_file():
        raise Pi05CometDataError(
            f"released Comet normalization asset is missing: {norm_file}"
        )
    return Behavior2026CometDataConfig(
        assets=_config.AssetsConfig(
            assets_dir=str(checkpoint / "assets"),
            asset_id=COMET_ASSET_ID,
        ),
        base_config=_config.DataConfig(
            prompt_from_task=True,
            behavior_dataset_root=str(Path(dataset_root).expanduser().resolve()),
            episodes_index=[int(value) for value in episode_indices],
        ),
    )


def _shape(feature: Any) -> tuple[int, ...]:
    values = feature["shape"] if isinstance(feature, dict) else feature.shape
    return tuple(int(value) for value in values)


def narrow_metadata_for_comet(meta: Any) -> None:
    """Select only the three RGB streams without changing the 61D state."""

    features = dict(meta.features)
    if _shape(features.get("observation.state", {})) != (RAW_STATE_DIM,):
        raise Pi05CometDataError("BEHAVIOR observation.state must remain raw 61D")
    if _shape(features.get("action", {})) != (ACTION_DIM,):
        raise Pi05CometDataError("BEHAVIOR action must be 23D")
    missing = [key for key in RGB_VIDEO_KEYS if key not in features]
    if missing:
        raise Pi05CometDataError(f"missing canonical RGB cameras: {missing}")
    selected = {
        "action": features["action"],
        "observation.state": features["observation.state"],
        **{key: features[key] for key in RGB_VIDEO_KEYS},
    }
    meta.info.features = selected


class LocalBehaviorWindowDataset:
    """Lazy read-only map dataset; memory use is independent of 207M rows."""

    def __init__(
        self,
        *,
        root: str | Path,
        sampling_manifest_path: str | Path,
        video_backend: str = "pyav",
        parquet_cache_files: int = 2,
    ) -> None:
        resolved_root = Path(root).expanduser().resolve()
        manifest_path = Path(sampling_manifest_path).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != "1.0"
            or payload.get("horizon") != ACTION_HORIZON
            or payload.get("io_contract") != "pi05_comet_lazy_parquet_rgb_v1"
        ):
            raise Pi05CometDataError(f"invalid PI0.5 lazy IO manifest: {manifest_path}")
        records = payload.get("episodes")
        if not isinstance(records, list) or not records:
            raise Pi05CometDataError("sampling manifest has no episodes")
        tasks: dict[int, str] = {}
        for line in (resolved_root / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                tasks[int(row["task_index"])] = str(row["task"])

        starts: list[int] = []
        stops: list[int] = []
        cursor = 0
        for record in records:
            length = int(record["length"])
            starts.append(cursor)
            cursor += length
            stops.append(cursor)
            if int(record["dataset_to_index"]) - int(record["dataset_from_index"]) != length:
                raise Pi05CometDataError(
                    f"episode {record['episode_index']} has inconsistent global bounds"
                )
            data_path = resolved_root / str(record["data_path"])
            video_paths = [
                resolved_root / str(record["videos"][key]["path"])
                for key in RGB_VIDEO_KEYS
            ]
            missing = [str(path) for path in (data_path, *video_paths) if not path.is_file()]
            if missing:
                raise Pi05CometDataError(
                    f"selected episode {record['episode_index']} is missing files: {missing}"
                )

        self.root = resolved_root
        self.manifest_path = manifest_path
        self.records = tuple(records)
        self.tasks = tasks
        self.video_backend = video_backend
        self._size = cursor
        self._stops = tuple(stops)
        self._parquet_cache_files = max(1, int(parquet_cache_files))
        self._parquet_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._video_pool: ThreadPoolExecutor | None = None
        self.episodes = tuple(int(record["episode_index"]) for record in records)
        self.episode_data_index = {"from": starts, "to": stops}
        self.obs_size = 1

    def __len__(self) -> int:
        return self._size

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_parquet_cache"] = OrderedDict()
        state["_video_pool"] = None
        return state

    def _load_parquet(self, relative_path: str) -> dict[str, np.ndarray]:
        cached = self._parquet_cache.pop(relative_path, None)
        if cached is not None:
            self._parquet_cache[relative_path] = cached
            return cached
        import pyarrow.parquet as pq

        table = pq.read_table(
            self.root / relative_path,
            columns=["index", "timestamp", "action", "observation.state"],
            memory_map=True,
        )
        indices = table["index"].combine_chunks().to_numpy(zero_copy_only=False)
        if len(indices) == 0 or int(indices[-1]) - int(indices[0]) + 1 != len(indices):
            raise Pi05CometDataError(f"non-contiguous index column in {relative_path}")
        values = {
            "first_index": np.asarray([int(indices[0])], dtype=np.int64),
            "timestamp": table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False),
            "action": table["action"]
            .combine_chunks()
            .values.to_numpy(zero_copy_only=False)
            .reshape(-1, ACTION_DIM),
            "state": table["observation.state"]
            .combine_chunks()
            .values.to_numpy(zero_copy_only=False)
            .reshape(-1, RAW_STATE_DIM),
        }
        self._parquet_cache[relative_path] = values
        while len(self._parquet_cache) > self._parquet_cache_files:
            self._parquet_cache.popitem(last=False)
        return values

    def _decode_video(self, relative_path: str, timestamp: float) -> torch.Tensor:
        from lerobot.datasets.video_utils import decode_video_frames

        return decode_video_frames(
            self.root / relative_path,
            [timestamp],
            tolerance_s=VIDEO_TOLERANCE_S,
            backend=self.video_backend,
            return_uint8=True,
            is_depth=False,
        ).squeeze(0)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        augmentation_seed: int | None = None
        if isinstance(index, (tuple, list)):
            if len(index) != 2:
                raise Pi05CometDataError("sampler index must be (frame, augmentation_seed)")
            index, augmentation_seed = int(index[0]), int(index[1])
        relative_index = int(index)
        if not 0 <= relative_index < self._size:
            raise IndexError(relative_index)
        episode_position = bisect.bisect_right(self._stops, relative_index)
        record = self.records[episode_position]
        episode_start = self.episode_data_index["from"][episode_position]
        local_frame = relative_index - episode_start
        if not (
            int(record["valid_from"])
            <= local_frame
            <= int(record["valid_to"]) - ACTION_HORIZON
        ):
            raise Pi05CometDataError(
                f"sampler emitted an invalid action window at relative frame {relative_index}"
            )
        global_index = int(record["dataset_from_index"]) + local_frame
        shard = self._load_parquet(str(record["data_path"]))
        row = global_index - int(shard["first_index"][0])
        if row < 0 or row + ACTION_HORIZON > len(shard["action"]):
            raise Pi05CometDataError(
                f"episode {record['episode_index']} crosses its declared parquet shard"
            )
        current_timestamp = float(shard["timestamp"][row])
        if self._video_pool is None:
            self._video_pool = ThreadPoolExecutor(max_workers=len(RGB_VIDEO_KEYS))
        futures = {
            key: self._video_pool.submit(
                self._decode_video,
                str(record["videos"][key]["path"]),
                float(record["videos"][key]["from_timestamp"]) + current_timestamp,
            )
            for key in RGB_VIDEO_KEYS
        }
        item: dict[str, Any] = {
            "observation.state": torch.from_numpy(shard["state"][row].copy()),
            "action": torch.from_numpy(
                shard["action"][row : row + ACTION_HORIZON].copy()
            ),
            "action_is_pad": torch.zeros(ACTION_HORIZON, dtype=torch.bool),
            "episode_index": torch.tensor(int(record["episode_index"])),
            "task_index": torch.tensor(int(record["task_index"])),
            "timestamp": torch.tensor(current_timestamp, dtype=torch.float32),
            "task": self.tasks[int(record["task_index"])],
            **{key: future.result() for key, future in futures.items()},
        }
        if augmentation_seed is not None:
            item["pi05_augmentation_seed"] = np.int64(augmentation_seed)
        return item


class ProcessShardedSampler(Sampler[tuple[int, int]]):
    """Shard one stateless global sample stream by JAX process and step."""

    def __init__(
        self,
        base_sampler: BudgetedResumableSampler,
        *,
        global_batch_size: int,
        process_count: int,
        process_index: int,
        start_step: int,
        total_steps: int,
    ) -> None:
        self.base_sampler = base_sampler
        self.global_batch_size = int(global_batch_size)
        self.process_count = int(process_count)
        self.process_index = int(process_index)
        self.start_step = int(start_step)
        self.total_steps = int(total_steps)
        if self.global_batch_size <= 0 or self.global_batch_size % self.process_count:
            raise ValueError("global batch must be positive and divisible by process_count")
        if not 0 <= self.process_index < self.process_count:
            raise ValueError("process_index is outside process_count")
        if not 0 <= self.start_step <= self.total_steps:
            raise ValueError("invalid resume step range")
        self.local_batch_size = self.global_batch_size // self.process_count

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for step in range(self.start_step, self.total_steps):
            start = (
                step * self.global_batch_size
                + self.process_index * self.local_batch_size
            )
            for counter in range(start, start + self.local_batch_size):
                yield self.base_sampler._sample(counter)  # noqa: SLF001

    def __len__(self) -> int:
        return (self.total_steps - self.start_step) * self.local_batch_size

    def state_dict(self) -> dict[str, Any]:
        manifest = Path(self.base_sampler.sampling_manifest_path)
        return {
            "schema_version": "1.0",
            "seed": self.base_sampler.seed,
            "strategy": self.base_sampler.strategy,
            "global_batch_size": self.global_batch_size,
            "process_count": self.process_count,
            "start_step": self.start_step,
            "total_steps": self.total_steps,
            "sampling_manifest": str(manifest.resolve()),
            "sampling_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }


class JaxBehaviorDataLoader:
    """PyTorch worker pool feeding process-local batches into a JAX sharding."""

    def __init__(
        self,
        *,
        data_config: _config.DataConfig,
        dataset: Any,
        sampler: ProcessShardedSampler,
        sharding: jax.sharding.Sharding,
        num_workers: int,
        prefetch_factor: int,
        persistent_workers: bool,
        seed: int,
    ) -> None:
        workers = int(num_workers)
        generator = torch.Generator().manual_seed(int(seed) + jax.process_index())
        kwargs: dict[str, Any] = {}
        if workers > 0:
            kwargs.update(
                multiprocessing_context=multiprocessing.get_context("spawn"),
                prefetch_factor=int(prefetch_factor),
                persistent_workers=bool(persistent_workers),
            )
        self._data_config = data_config
        self.sampler = sampler
        self.torch_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=sampler.local_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=workers,
            collate_fn=_comet_collate,
            worker_init_fn=_data_loader._worker_init_fn,  # noqa: SLF001
            drop_last=True,
            generator=generator,
            **kwargs,
        )
        self._sharding = sharding

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self.torch_loader:
            global_batch = jax.tree.map(
                lambda value: jax.make_array_from_process_local_data(
                    self._sharding, value
                ),
                batch,
            )
            yield _model.Observation.from_dict(global_batch), global_batch["actions"]


def _comet_collate(items: Sequence[Any]) -> Any:
    """Stack with model-native dtypes before host-to-device transfer."""

    def stack(*values: Any) -> np.ndarray:
        result = np.stack([np.asarray(value) for value in values], axis=0)
        if np.issubdtype(result.dtype, np.floating) and result.dtype != np.float32:
            return result.astype(np.float32)
        if np.issubdtype(result.dtype, np.signedinteger) and result.dtype != np.int32:
            return result.astype(np.int32)
        return result

    return jax.tree.map(stack, *items)


def create_data_loader(
    config: _config.TrainConfig,
    *,
    episode_indices: Sequence[int],
    episode_lengths: Sequence[int],
    sampling_manifest_path: str | Path,
    sharding: jax.sharding.Sharding,
    start_step: int,
    num_workers: int | None = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    natural_probability: float = 0.70,
    skill_probability: float = 0.20,
    boundary_probability: float = 0.10,
    task_reuse_steps: int = 2,
    episode_reuse_steps: int = 2,
    window_batch_locality_span: int = 96,
) -> JaxBehaviorDataLoader:
    if not isinstance(config.data, Behavior2026CometDataConfig):
        raise Pi05CometDataError("unexpected Comet data factory")
    data_config = config.data.create(config.assets_dirs, config.model)
    raw = LocalBehaviorWindowDataset(
        root=data_config.behavior_dataset_root,
        sampling_manifest_path=sampling_manifest_path,
    )
    prompted = _data_loader.TransformedDataset(
        raw,
        [_transforms.PromptFromLeRobotItem()],
    )
    transformed = _data_loader.transform_dataset(prompted, data_config)

    process_count = jax.process_count()
    process_index = jax.process_index()
    local_batch = config.batch_size // process_count
    base_sampler = BudgetedResumableSampler(
        raw,
        seed=config.seed,
        batch_size=local_batch,
        num_processes=process_count,
        strategy="task_hierarchical",
        samples_per_epoch=config.num_train_steps * config.batch_size,
        drop_padded_windows=True,
        sampling_manifest_path=str(Path(sampling_manifest_path).resolve()),
        natural_probability=natural_probability,
        skill_probability=skill_probability,
        boundary_probability=boundary_probability,
        task_reuse_steps=task_reuse_steps,
        episode_batch_locality=True,
        episode_reuse_steps=episode_reuse_steps,
        window_batch_locality_span=window_batch_locality_span,
    )
    sampler = ProcessShardedSampler(
        base_sampler,
        global_batch_size=config.batch_size,
        process_count=process_count,
        process_index=process_index,
        start_step=start_step,
        total_steps=config.num_train_steps,
    )
    logging.info(
        "Behavior loader process=%d/%d local_batch=%d start_step=%d samples=%d",
        process_index,
        process_count,
        local_batch,
        start_step,
        len(sampler),
    )
    return JaxBehaviorDataLoader(
        data_config=data_config,
        dataset=transformed,
        sampler=sampler,
        sharding=sharding,
        num_workers=config.num_workers if num_workers is None else int(num_workers),
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        seed=config.seed,
    )


def load_episode_selection(path: str | Path) -> tuple[list[int], list[int]]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise Pi05CometDataError(f"invalid sampling manifest: {path}")
    return (
        [int(row["episode_index"]) for row in episodes],
        [int(row["length"]) for row in episodes],
    )
