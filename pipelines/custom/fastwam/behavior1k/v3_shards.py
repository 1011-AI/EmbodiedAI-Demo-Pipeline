"""Standalone LeRobot v3 shard compatibility helpers for pinned FastWAM.

The FastWAM workspace vendors a LeRobot v2.1-era loader, while the
BEHAVIOR-1K 2026 challenge dataset uses the v3 layout:

* episode metadata is stored in ``meta/episodes/chunk-*/file-*.parquet``;
* multiple episodes share one data Parquet and one video MP4;
* video timestamps are episode-local and must be shifted by the episode's
  ``from_timestamp`` before decoding the shared MP4.

This file deliberately has no imports from the demo repository.  It is copied
verbatim into the generated FastWAM source tree so the offline GPU node only
needs the pinned FastWAM environment.
"""

from __future__ import annotations

import bisect
from collections import Counter, OrderedDict
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


V3_VERSION_PREFIX = "v3."


def is_lerobot_v3(info: Mapping[str, Any]) -> bool:
    return str(info.get("codebase_version", "")).startswith(V3_VERSION_PREFIX)


def _as_int(value: Any) -> int:
    if hasattr(value, "as_py"):
        value = value.as_py()
    elif hasattr(value, "item"):
        value = value.item()
    return int(value)


def _dataset_info(root: Path) -> dict[str, Any]:
    path = root / "meta/info.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing LeRobot metadata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not is_lerobot_v3(payload):
        raise ValueError(
            f"expected LeRobot v3 meta/info.json at {path}, "
            f"got codebase_version={payload.get('codebase_version')!r}"
        )
    return payload


def _required_episode_columns(info: Mapping[str, Any]) -> set[str]:
    required = {
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    }
    for key, feature in (info.get("features") or {}).items():
        if (feature or {}).get("dtype") != "video":
            continue
        required.update(
            {
                f"videos/{key}/chunk_index",
                f"videos/{key}/file_index",
                f"videos/{key}/from_timestamp",
                f"videos/{key}/to_timestamp",
            }
        )
    return required


def load_v3_episode_metadata(root: str | Path) -> dict[int, dict[str, Any]]:
    """Load and strictly validate the real v3 episode Parquet metadata."""

    root = Path(root).expanduser().resolve()
    info = _dataset_info(root)
    episode_files = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(
            f"no LeRobot v3 episode metadata under {root / 'meta/episodes'}"
        )
    try:
        import pyarrow.dataset as pa_dataset
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to read LeRobot v3 episode metadata"
        ) from exc

    episode_dataset = pa_dataset.dataset(
        [str(path) for path in episode_files],
        format="parquet",
    )
    available = set(episode_dataset.schema.names)
    required = _required_episode_columns(info)
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "LeRobot v3 episode metadata is missing required flattened columns: "
            f"{missing}"
        )

    table = episode_dataset.to_table()
    rows: dict[int, dict[str, Any]] = {}
    for raw_row in table.to_pylist():
        row = dict(raw_row)
        episode_index = _as_int(row["episode_index"])
        if episode_index in rows:
            raise ValueError(
                f"duplicate episode_index={episode_index} in LeRobot v3 metadata"
            )
        length = _as_int(row["length"])
        dataset_from = _as_int(row["dataset_from_index"])
        dataset_to = _as_int(row["dataset_to_index"])
        if length <= 0 or dataset_to - dataset_from != length:
            raise ValueError(
                f"invalid v3 episode bounds for episode {episode_index}: "
                f"length={length}, from={dataset_from}, to={dataset_to}"
            )
        rows[episode_index] = row

    expected_total = _as_int(info["total_episodes"])
    if len(rows) != expected_total:
        raise ValueError(
            f"LeRobot v3 episode metadata count mismatch: "
            f"expected={expected_total}, actual={len(rows)}"
        )
    ordered = sorted(rows)
    if ordered != list(range(expected_total)):
        raise ValueError(
            "LeRobot v3 episode indices must be contiguous 0..total_episodes-1; "
            f"first={ordered[:5]}, last={ordered[-5:]}"
        )
    return {episode_index: rows[episode_index] for episode_index in ordered}


def v3_data_file_path(
    info: Mapping[str, Any],
    episode: Mapping[str, Any],
) -> Path:
    template = str(info["data_path"])
    return Path(
        template.format(
            chunk_index=_as_int(episode["data/chunk_index"]),
            file_index=_as_int(episode["data/file_index"]),
        )
    )


def v3_video_file_path(
    info: Mapping[str, Any],
    episode: Mapping[str, Any],
    video_key: str,
) -> Path:
    template = info.get("video_path")
    if not template:
        raise ValueError("LeRobot v3 dataset has no video_path template")
    return Path(
        str(template).format(
            video_key=video_key,
            chunk_index=_as_int(episode[f"videos/{video_key}/chunk_index"]),
            file_index=_as_int(episode[f"videos/{video_key}/file_index"]),
        )
    )


def _episode_values(values: Iterable[Any]) -> Iterable[int]:
    for value in values:
        yield _as_int(value)


def filter_v3_hf_dataset(
    hf_dataset: Any,
    selected_episodes: Sequence[int],
    episode_metadata: Mapping[int, Mapping[str, Any]],
) -> Any:
    """Filter shared Parquet shards to the exact selected episode rows.

    The function validates membership, order, and per-episode frame counts
    after filtering.  Loading only the unique shard paths is insufficient:
    every shard can contain multiple tasks and episodes.
    """

    selected = [_as_int(value) for value in selected_episodes]
    if not selected:
        raise ValueError("selected_episodes must not be empty")
    if len(selected) != len(set(selected)):
        raise ValueError("selected_episodes must contain unique episode indices")
    missing_metadata = [index for index in selected if index not in episode_metadata]
    if missing_metadata:
        raise ValueError(
            f"selected episodes are absent from v3 metadata: {missing_metadata[:10]}"
        )

    selected_set = frozenset(selected)

    def _keep(batch: Sequence[Any]) -> list[bool]:
        return [_as_int(value) in selected_set for value in batch]

    filtered = hf_dataset.filter(
        _keep,
        input_columns=["episode_index"],
        batched=True,
        desc="Selecting exact LeRobot v3 episodes",
    )

    counts: Counter[int] = Counter()
    observed_order: list[int] = []
    previous: int | None = None
    for episode_index in _episode_values(filtered["episode_index"]):
        counts[episode_index] += 1
        if episode_index != previous:
            observed_order.append(episode_index)
            previous = episode_index

    if observed_order != selected:
        raise ValueError(
            "filtered v3 rows are not grouped in selected episode order: "
            f"expected={selected[:10]}, observed={observed_order[:10]}"
        )
    unexpected = sorted(set(counts) - selected_set)
    if unexpected:
        raise ValueError(f"filtered v3 rows contain unselected episodes: {unexpected[:10]}")
    for episode_index in selected:
        expected = _as_int(episode_metadata[episode_index]["length"])
        actual = counts.get(episode_index, 0)
        if actual != expected:
            raise ValueError(
                f"v3 frame count mismatch for episode {episode_index}: "
                f"expected={expected}, actual={actual}"
            )
    return filtered


def read_v3_episode_table(
    path: str | Path,
    episode_index: int,
    episode: Mapping[str, Any],
) -> Any:
    """Read exactly one episode from a shared v3 data Parquet."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ImportError("pyarrow is required to read LeRobot v3 data") from exc
    episode_index = _as_int(episode_index)
    table = parquet.read_table(
        str(path),
        filters=[("episode_index", "=", episode_index)],
    )
    expected = _as_int(episode["length"])
    if table.num_rows != expected:
        raise ValueError(
            f"shared v3 Parquet did not yield exactly episode {episode_index}: "
            f"expected={expected}, actual={table.num_rows}, path={path}"
        )
    observed = {_as_int(value) for value in table["episode_index"].to_pylist()}
    if observed != {episode_index}:
        raise ValueError(
            f"shared v3 Parquet mixed episode rows: expected={episode_index}, "
            f"observed={sorted(observed)}"
        )
    return table


class _LazyV3Selection:
    """Small ``datasets.Dataset.select`` compatible view.

    FastWAM only selects the 9/32/33 indices belonging to one training
    window.  Keeping this view lazy avoids materializing an Arrow dataset for
    the full 210M-frame corpus.
    """

    def __init__(self, dataset: "LazyV3ParquetDataset", indices: Sequence[int]) -> None:
        self._dataset = dataset
        self._indices = tuple(_as_int(index) for index in indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, key: str) -> list[Any]:
        if not isinstance(key, str):
            raise TypeError("lazy v3 selections only support column-name access")
        return self._dataset.column(key, self._indices)


class LazyV3ParquetDataset:
    """Random-access LeRobot v3 data without a per-rank 70 GiB HF cache.

    LeRobot v3 groups roughly one hundred episodes into each Parquet file.
    Loading all selected files through ``datasets.load_dataset`` makes every
    distributed rank scan and materialize the full corpus before training.
    This adapter instead maps the selected dataset's local frame indices to
    their shared shard, loads a bounded number of complete shards, and slices
    exact episode rows using the authoritative v3 metadata.

    A complete shard is cached rather than repeatedly applying an
    ``episode_index`` Parquet filter because the BEHAVIOR-1K files contain a
    single row group.  The default three-shard cache matches the bounded task
    reuse sampler (one task normally spans two or three shards) and is local
    to each DataLoader worker.  Override it with
    ``FASTWAM_V3_SHARD_CACHE_SIZE`` after measuring host-memory pressure.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        info: Mapping[str, Any],
        selected_episodes: Sequence[int],
        episode_metadata: Mapping[int, Mapping[str, Any]],
        features: Any = None,
        cache_size: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.info = dict(info)
        self.features = features
        self.selected_episodes = tuple(_as_int(value) for value in selected_episodes)
        if not self.selected_episodes:
            raise ValueError("selected_episodes must not be empty")
        if len(self.selected_episodes) != len(set(self.selected_episodes)):
            raise ValueError("selected_episodes must contain unique episode indices")
        missing = [value for value in self.selected_episodes if value not in episode_metadata]
        if missing:
            raise ValueError(f"selected episodes are absent from v3 metadata: {missing[:10]}")

        if cache_size is None:
            raw_cache_size = os.environ.get("FASTWAM_V3_SHARD_CACHE_SIZE", "3")
            try:
                cache_size = int(raw_cache_size)
            except ValueError as exc:
                raise ValueError(
                    "FASTWAM_V3_SHARD_CACHE_SIZE must be an integer, "
                    f"got {raw_cache_size!r}"
                ) from exc
        self.cache_size = int(cache_size)
        if self.cache_size <= 0:
            raise ValueError("v3 shard cache_size must be positive")

        mandatory = {"episode_index", "task_index", "timestamp"}
        feature_columns = {
            key
            for key, feature in (self.info.get("features") or {}).items()
            if (feature or {}).get("dtype") != "video"
        }
        self.column_names = tuple(sorted(mandatory | feature_columns))

        # ``dataset_from_index`` is global to the corpus.  Derive the base and
        # expected row count of every shared file once from the small episode
        # metadata table, then map selected-local indices via cumulative
        # episode lengths.
        shard_bounds: dict[Path, list[int]] = {}
        for episode in episode_metadata.values():
            path = self.root / v3_data_file_path(self.info, episode)
            start = _as_int(episode["dataset_from_index"])
            stop = _as_int(episode["dataset_to_index"])
            bounds = shard_bounds.setdefault(path, [start, stop])
            bounds[0] = min(bounds[0], start)
            bounds[1] = max(bounds[1], stop)
        self._shard_bounds = {
            path: (bounds[0], bounds[1]) for path, bounds in shard_bounds.items()
        }

        cumulative = 0
        self._cumulative_stops: list[int] = []
        self._episode_locations: list[tuple[int, Path, int, int]] = []
        for episode_index in self.selected_episodes:
            episode = episode_metadata[episode_index]
            path = self.root / v3_data_file_path(self.info, episode)
            global_start = _as_int(episode["dataset_from_index"])
            length = _as_int(episode["length"])
            shard_start = self._shard_bounds[path][0]
            local_shard_start = global_start - shard_start
            self._episode_locations.append(
                (episode_index, path, local_shard_start, length)
            )
            cumulative += length
            self._cumulative_stops.append(cumulative)
        self._length = cumulative
        self._cache: OrderedDict[Path, Any] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def __getstate__(self) -> dict[str, Any]:
        # Spawned DataLoader workers must never pickle large Arrow tables from
        # a parent-side smoke read.  Every worker owns a bounded independent
        # cache while the OS page cache remains shared by the node.
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        state["cache_hits"] = 0
        state["cache_misses"] = 0
        return state

    def __len__(self) -> int:
        return self._length

    def set_transform(self, _transform: Any) -> None:
        # Values are converted to torch tensors in ``_value``.  This method is
        # present solely for compatibility with the pinned LeRobot loader.
        return None

    def select(self, indices: Sequence[int]) -> _LazyV3Selection:
        return _LazyV3Selection(self, indices)

    def _location(self, index: int) -> tuple[Path, int, int]:
        index = _as_int(index)
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(f"lazy v3 frame index {index} out of range {self._length}")
        position = bisect.bisect_right(self._cumulative_stops, index)
        episode_local_base = 0 if position == 0 else self._cumulative_stops[position - 1]
        episode_index, path, shard_episode_start, length = self._episode_locations[position]
        offset = index - episode_local_base
        if not 0 <= offset < length:
            raise RuntimeError(
                f"invalid lazy v3 episode offset: episode={episode_index}, "
                f"offset={offset}, length={length}"
            )
        return path, shard_episode_start + offset, episode_index

    def _load_shard(self, path: Path) -> Any:
        table = self._cache.pop(path, None)
        if table is not None:
            self.cache_hits += 1
            self._cache[path] = table
            return table
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ImportError("pyarrow is required for lazy LeRobot v3 loading") from exc
        if not path.is_file():
            raise FileNotFoundError(f"missing LeRobot v3 data shard: {path}")
        table = parquet.read_table(str(path), columns=list(self.column_names))
        shard_start, shard_stop = self._shard_bounds[path]
        expected_rows = shard_stop - shard_start
        if table.num_rows != expected_rows:
            raise ValueError(
                f"v3 shared shard row count mismatch for {path}: "
                f"expected={expected_rows}, actual={table.num_rows}"
            )
        self.cache_misses += 1
        self._cache[path] = table
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return table

    @staticmethod
    def _to_torch(value: Any) -> Any:
        if hasattr(value, "as_py"):
            value = value.as_py()
        if value is None or isinstance(value, str):
            return value
        return torch.tensor(value)

    def _value(self, key: str, index: int) -> Any:
        if key not in self.column_names:
            raise KeyError(key)
        path, shard_row, expected_episode = self._location(index)
        table = self._load_shard(path)
        value = table[key][shard_row]
        if key == "episode_index" and _as_int(value) != expected_episode:
            raise ValueError(
                f"v3 shard index mapping leaked episode rows at local index {index}: "
                f"expected={expected_episode}, actual={_as_int(value)}, path={path}"
            )
        return self._to_torch(value)

    def column(self, key: str, indices: Sequence[int]) -> list[Any]:
        return [self._value(key, index) for index in indices]

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, str):
            raise TypeError(
                "full-column reads are disabled for lazy v3 datasets; use select(indices)[column]"
            )
        if isinstance(index, slice):
            indices = range(*index.indices(self._length))
            return {
                key: self.column(key, indices)
                for key in self.column_names
            }
        if isinstance(index, Sequence) and not isinstance(index, (str, bytes)):
            return {
                key: self.column(key, index)
                for key in self.column_names
            }
        return {key: self._value(key, index) for key in self.column_names}


def shift_v3_video_timestamps(
    episode: Mapping[str, Any],
    video_key: str,
    query_timestamps: Sequence[float],
) -> list[float]:
    """Shift episode-local timestamps into the shared v3 MP4 timeline."""

    offset = float(episode[f"videos/{video_key}/from_timestamp"])
    upper = float(episode[f"videos/{video_key}/to_timestamp"])
    shifted = [offset + float(timestamp) for timestamp in query_timestamps]
    tolerance = 1e-6
    if any(timestamp < offset - tolerance or timestamp > upper + tolerance for timestamp in shifted):
        raise ValueError(
            f"video timestamps escape episode bounds for {video_key}: "
            f"offset={offset}, upper={upper}, shifted={shifted[:5]}"
        )
    return shifted
