from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_demo.behavior1k.r1pro import ACTION_DIM, POLICY_STATE_DIM, RAW_STATE_DIM
from embodied_demo.behavior1k.schemas import EpisodeReference
from embodied_demo.errors import ConfigurationError, SchemaValidationError

QUANTILES: tuple[tuple[str, float], ...] = (
    ("q01", 0.01),
    ("q10", 0.10),
    ("q50", 0.50),
    ("q90", 0.90),
    ("q99", 0.99),
)
DEFAULT_BATCH_SIZE = 65_536
DEFAULT_MAX_QUANTILE_ROWS = 100_000


@dataclass(frozen=True)
class ComputedBehaviorStats:
    """Statistics produced from one exact set of selected dataset episodes."""

    policy_stats: dict[str, dict[str, Any]]
    raw_state_stats: dict[str, dict[str, Any]]
    frame_count: int
    quantile_method: str
    quantile_sample_count: int
    quantile_sample_limit: int


class _RunningVectorStats:
    """Numerically stable, bounded-memory moments for fixed-width vectors."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.count = 0
        self.minimum: Any = None
        self.maximum: Any = None
        self.mean: Any = None
        self.m2: Any = None

    def update(self, values: Any) -> None:
        import numpy as np

        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise SchemaValidationError(
                f"statistics input must have shape [N, {self.dimension}], got {matrix.shape}"
            )
        if matrix.shape[0] == 0:
            return
        if not np.isfinite(matrix).all():
            raise SchemaValidationError("statistics input contains non-finite values")

        batch_count = int(matrix.shape[0])
        batch_minimum = matrix.min(axis=0)
        batch_maximum = matrix.max(axis=0)
        batch_mean = matrix.mean(axis=0, dtype=np.float64)
        centered = matrix - batch_mean
        batch_m2 = np.square(centered).sum(axis=0, dtype=np.float64)

        if self.count == 0:
            self.count = batch_count
            self.minimum = batch_minimum
            self.maximum = batch_maximum
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        combined_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / combined_count)
        self.m2 = (
            self.m2
            + batch_m2
            + np.square(delta) * self.count * batch_count / combined_count
        )
        self.minimum = np.minimum(self.minimum, batch_minimum)
        self.maximum = np.maximum(self.maximum, batch_maximum)
        self.count = combined_count

    def result(self, quantile_values: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        if self.count == 0:
            raise SchemaValidationError("cannot finalize empty statistics")
        result: dict[str, Any] = {
            # LeRobot stores the reduced frame count as a one-element vector.
            "count": [self.count],
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "mean": self.mean.tolist(),
            "std": np.sqrt(self.m2 / self.count).tolist(),
        }
        result.update(
            {
                key: np.asarray(value).tolist()
                for key, value in quantile_values.items()
            }
        )
        return result


class _SharedPriorityReservoir:
    """Uniformly sample aligned rows using deterministic random priorities.

    Exact min/max/moments are accumulated separately. Only quantiles use this
    bounded reservoir when a selected view contains more than ``limit`` rows.
    """

    def __init__(self, *, limit: int, seed: int = 2026) -> None:
        import numpy as np

        if limit <= 0:
            raise ValueError("quantile sample limit must be positive")
        self.limit = limit
        self._rng = np.random.default_rng(seed)
        self._priorities = np.empty((0,), dtype=np.float64)
        self._values: dict[str, Any] = {}
        self.total_rows = 0

    @property
    def sample_count(self) -> int:
        return int(self._priorities.shape[0])

    def update(self, values: Mapping[str, Any]) -> None:
        import numpy as np

        if not values:
            return
        matrices = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in values.items()
        }
        row_counts = {matrix.shape[0] for matrix in matrices.values()}
        if len(row_counts) != 1:
            raise SchemaValidationError("aligned quantile inputs have different row counts")
        row_count = row_counts.pop()
        if row_count == 0:
            return
        if any(matrix.ndim != 2 for matrix in matrices.values()):
            raise SchemaValidationError("quantile inputs must be two-dimensional")

        priorities = self._rng.random(row_count)
        if not self._values:
            self._values = {
                key: np.empty((0, matrix.shape[1]), dtype=np.float64)
                for key, matrix in matrices.items()
            }
        elif set(matrices) != set(self._values):
            raise SchemaValidationError("quantile input keys changed while accumulating statistics")

        combined_priorities = np.concatenate((self._priorities, priorities))
        combined_values = {
            key: np.concatenate((self._values[key], matrix), axis=0)
            for key, matrix in matrices.items()
        }
        if combined_priorities.shape[0] > self.limit:
            selected = np.argpartition(combined_priorities, -self.limit)[-self.limit :]
            # Stable ordering is not required for quantiles, but sorting the
            # selected indices makes repeated runs byte-for-byte reproducible.
            selected.sort()
            combined_priorities = combined_priorities[selected]
            combined_values = {
                key: matrix[selected]
                for key, matrix in combined_values.items()
            }

        self._priorities = combined_priorities
        self._values = combined_values
        self.total_rows += row_count

    def quantiles(self, key: str) -> dict[str, Any]:
        import numpy as np

        if key not in self._values or self.sample_count == 0:
            raise SchemaValidationError(f"no quantile samples accumulated for {key}")
        matrix = self._values[key]
        return {
            name: np.quantile(matrix, probability, axis=0)
            for name, probability in QUANTILES
        }


def _vector_column_to_numpy(column: Any, *, dimension: int, name: str) -> Any:
    """Convert an Arrow list column to one dense NumPy matrix per lazy batch."""

    import numpy as np

    try:
        flattened = column.flatten()
        values = flattened.to_numpy(zero_copy_only=False)
        matrix = np.asarray(values, dtype=np.float64).reshape(len(column), dimension)
    except (AttributeError, TypeError, ValueError):
        # This fallback handles uncommon Arrow extension/list representations.
        matrix = np.asarray(column.to_pylist(), dtype=np.float64)
    if matrix.shape != (len(column), dimension):
        raise SchemaValidationError(
            f"{name} must have shape [N, {dimension}], got {matrix.shape}"
        )
    return matrix


def _project_policy_state_batch(raw_state: Any) -> Any:
    """Vectorized form of the canonical R1Pro raw61-to-policy23 projection."""

    import numpy as np

    projected = np.concatenate(
        (
            raw_state[:, 0:3],
            raw_state[:, 53:57],
            raw_state[:, 3:10],
            (raw_state[:, 24] + raw_state[:, 25])[:, None],
            raw_state[:, 28:35],
            (raw_state[:, 49] + raw_state[:, 50])[:, None],
        ),
        axis=1,
    )
    if projected.shape[1] != POLICY_STATE_DIM:  # pragma: no cover - edit guard.
        raise AssertionError(
            f"internal R1Pro state projection produced {projected.shape[1]} values"
        )
    return projected


def _group_episodes_by_data_path(
    episodes: Sequence[EpisodeReference],
) -> dict[str, list[EpisodeReference]]:
    grouped: dict[str, list[EpisodeReference]] = defaultdict(list)
    seen: set[int] = set()
    for episode in episodes:
        if episode.episode_index in seen:
            raise SchemaValidationError(
                f"selected episodes contain duplicate episode_index={episode.episode_index}"
            )
        seen.add(episode.episode_index)
        grouped[episode.data.relative_path].append(episode)
    if not grouped:
        raise SchemaValidationError("cannot compute statistics for an empty episode selection")
    return dict(grouped)


def compute_selected_episode_stats(
    *,
    source_root: Path,
    episodes: Sequence[EpisodeReference],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_quantile_rows: int = DEFAULT_MAX_QUANTILE_ROWS,
) -> ComputedBehaviorStats:
    """Compute train-time statistics without decoding or opening any video.

    Parquet shards are scanned once in lazy record batches. Every batch is
    filtered by ``episode_index`` before action/state values are accumulated.
    Exact min/max/mean/population-std are retained for all selected frames.
    Quantiles are exact up to ``max_quantile_rows`` and otherwise use a
    deterministic uniform row reservoir whose size and method are recorded in
    the view manifest.
    """

    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ConfigurationError(
            "Behavior statistics require NumPy and PyArrow. "
            "Install the Behavior extras with: pip install -e '.[behavior1k]'"
        ) from exc

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    grouped = _group_episodes_by_data_path(episodes)
    expected_lengths = {episode.episode_index: episode.length for episode in episodes}
    observed_lengths: dict[int, int] = defaultdict(int)

    action_stats = _RunningVectorStats(ACTION_DIM)
    raw_state_stats = _RunningVectorStats(RAW_STATE_DIM)
    policy_state_stats = _RunningVectorStats(POLICY_STATE_DIM)
    reservoir = _SharedPriorityReservoir(limit=max_quantile_rows)

    for relative_path, shard_episodes in sorted(grouped.items()):
        path = source_root / relative_path
        if not path.is_file():
            raise ConfigurationError(f"selected data Parquet file does not exist: {path}")
        parquet_file = pq.ParquetFile(path)
        available_columns = set(parquet_file.schema_arrow.names)
        required_columns = {"episode_index", "action", "observation.state"}
        missing_columns = sorted(required_columns - available_columns)
        if missing_columns:
            raise SchemaValidationError(
                f"selected data shard {path} is missing columns: {missing_columns}"
            )

        selected_ids = np.asarray(
            [episode.episode_index for episode in shard_episodes],
            dtype=np.int64,
        )
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["episode_index", "action", "observation.state"],
            use_threads=True,
        ):
            episode_ids = np.asarray(
                batch.column("episode_index").to_numpy(zero_copy_only=False),
                dtype=np.int64,
            )
            selected_mask = np.isin(episode_ids, selected_ids)
            if not selected_mask.any():
                continue

            selected_episode_ids = episode_ids[selected_mask]
            unique_ids, counts = np.unique(selected_episode_ids, return_counts=True)
            for episode_id, count in zip(unique_ids.tolist(), counts.tolist(), strict=True):
                observed_lengths[int(episode_id)] += int(count)

            action = _vector_column_to_numpy(
                batch.column("action"),
                dimension=ACTION_DIM,
                name="action",
            )[selected_mask]
            raw_state = _vector_column_to_numpy(
                batch.column("observation.state"),
                dimension=RAW_STATE_DIM,
                name="observation.state",
            )[selected_mask]
            policy_state = _project_policy_state_batch(raw_state)

            action_stats.update(action)
            raw_state_stats.update(raw_state)
            policy_state_stats.update(policy_state)
            reservoir.update(
                {
                    "action": action,
                    "raw_state": raw_state,
                    "policy_state": policy_state,
                }
            )

    mismatches = {
        episode_index: {
            "expected": expected,
            "observed": observed_lengths.get(episode_index, 0),
        }
        for episode_index, expected in sorted(expected_lengths.items())
        if observed_lengths.get(episode_index, 0) != expected
    }
    if mismatches:
        preview = dict(list(mismatches.items())[:20])
        raise SchemaValidationError(
            "selected episode frame counts do not match episode metadata: "
            f"{preview}"
        )

    quantile_method = (
        "exact"
        if reservoir.total_rows <= reservoir.limit
        else "deterministic_priority_reservoir"
    )
    policy_stats = {
        "action": action_stats.result(reservoir.quantiles("action")),
        "observation.state": policy_state_stats.result(
            reservoir.quantiles("policy_state")
        ),
    }
    audit_stats = {
        "observation.state": raw_state_stats.result(
            reservoir.quantiles("raw_state")
        )
    }
    return ComputedBehaviorStats(
        policy_stats=policy_stats,
        raw_state_stats=audit_stats,
        frame_count=action_stats.count,
        quantile_method=quantile_method,
        quantile_sample_count=reservoir.sample_count,
        quantile_sample_limit=reservoir.limit,
    )
