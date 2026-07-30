from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from embodied_demo.behavior1k.r1pro import DEPTH_VIDEO_KEYS, RGB_VIDEO_KEYS
from embodied_demo.behavior1k.schemas import EpisodeReference
from embodied_demo.errors import ConfigurationError, SchemaValidationError

MaterializationMode = Literal["hardlink", "copy"]
MATERIALIZATION_MANIFEST = "materialization_manifest.json"


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"required JSON file not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"expected a JSON object in {path}")
    return payload


def _safe_relative_path(raw_path: str, *, field: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SchemaValidationError(f"{field} must be a safe relative path, got {raw_path!r}")
    return relative


def _load_episodes(path: Path) -> list[EpisodeReference]:
    episodes: list[EpisodeReference] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigurationError(f"cannot read episode manifest {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            episodes.append(EpisodeReference.model_validate(payload))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise SchemaValidationError(
                f"invalid episode reference at {path}:{line_number}: {exc}"
            ) from exc
    if not episodes:
        raise SchemaValidationError(f"episode manifest is empty: {path}")
    indices = [episode.episode_index for episode in episodes]
    if len(indices) != len(set(indices)):
        raise SchemaValidationError(f"episode manifest contains duplicate episode_index values: {path}")
    return episodes


def _resolve_source_root(
    manifest: Mapping[str, object],
    source_root_override: str | Path | None,
) -> Path:
    raw_root = source_root_override if source_root_override is not None else manifest.get("source_root")
    if raw_root is None or not str(raw_root).strip():
        raise SchemaValidationError("view_manifest.json does not define source_root")
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(f"BEHAVIOR-1K source root does not exist or is not a directory: {root}")
    return root


def _narrow_info(source: Path, destination: Path) -> None:
    info = _read_json_object(source)
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise SchemaValidationError(f"{source} does not contain a feature mapping")
    missing_rgb = [key for key in RGB_VIDEO_KEYS if key not in features]
    if missing_rgb:
        raise SchemaValidationError(f"{source} is missing canonical RGB features: {missing_rgb}")
    narrowed_features: dict[str, object] = {}
    for key, value in features.items():
        feature = value if isinstance(value, Mapping) else {}
        is_video = feature.get("dtype") == "video"
        if key in DEPTH_VIDEO_KEYS or str(key).startswith("observation.depth"):
            continue
        if is_video and key not in RGB_VIDEO_KEYS:
            continue
        narrowed_features[str(key)] = value
    info["features"] = narrowed_features
    _atomic_write_json(destination, info)


def _narrow_stats(source: Path, destination: Path) -> None:
    stats = _read_json_object(source)
    narrowed = {
        str(key): value
        for key, value in stats.items()
        if key not in DEPTH_VIDEO_KEYS and not str(key).startswith("observation.depth")
    }
    _atomic_write_json(destination, narrowed)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy_or_link(
    source: Path,
    destination: Path,
    *,
    mode: MaterializationMode,
) -> tuple[int, bool]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if mode == "hardlink":
            os.link(source, destination, follow_symlinks=True)
        else:
            shutil.copy2(source, destination, follow_symlinks=True)
    except FileExistsError as exc:
        raise ConfigurationError(
            f"materialization target already exists: {destination}; use a new empty output root"
        ) from exc
    except OSError as exc:
        if mode == "hardlink" and exc.errno in {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.ENOTSUP,
        }:
            raise ConfigurationError(
                "hardlink materialization failed; the source and output must be on the same "
                f"filesystem and allow hard links: {source} -> {destination}: {exc}. "
                "No copy fallback was attempted. Re-run explicitly with --mode copy if the "
                "additional storage use is acceptable."
            ) from exc
        raise ConfigurationError(
            f"cannot materialize {source} -> {destination}: {exc}"
        ) from exc
    source_stat = source.stat()
    destination_stat = destination.stat()
    inode_reused = (
        source_stat.st_dev == destination_stat.st_dev
        and source_stat.st_ino == destination_stat.st_ino
    )
    if mode == "hardlink" and not inode_reused:
        raise ConfigurationError(
            f"hardlink verification failed for {source} -> {destination}"
        )
    return destination_stat.st_size, inode_reused


def _prepare_output_parent(output_root: Path) -> None:
    if output_root.exists() and not output_root.is_dir():
        raise ConfigurationError(f"output root exists and is not a directory: {output_root}")
    if output_root.is_dir() and any(output_root.iterdir()):
        raise ConfigurationError(
            f"output root must be absent or empty to prevent stale dataset files: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    # A sibling staging directory is renamed into place only after every source
    # file and the manifest have been written.  Removing a caller-provided empty
    # directory is safe and keeps the final rename atomic on one filesystem.
    if output_root.is_dir():
        output_root.rmdir()


def _validate_source_revision(source_root: Path, expected_revision: str) -> None:
    markers = (
        source_root / ".dataset_revision",
        source_root / "REVISION",
        source_root / ".huggingface/revision",
    )
    marker = next((path for path in markers if path.is_file()), None)
    if marker is None:
        raise SchemaValidationError(
            "cannot verify the pinned BEHAVIOR-1K source revision because no revision "
            f"marker exists under {source_root}; expected {expected_revision}"
        )
    actual = marker.read_text(encoding="utf-8").strip()
    if actual != expected_revision:
        raise SchemaValidationError(
            f"source revision marker mismatch: expected {expected_revision}, got {actual}"
        )


def _build_source_plan(
    *,
    source_root: Path,
    episodes: list[EpisodeReference],
) -> dict[str, str]:
    """Return relative source paths mapped to inventory categories.

    All episode metadata Parquets are intentionally retained. LeRobot v3 indexes
    that table by the global ``episode_index``; keeping the tiny metadata table
    avoids rewriting or guessing episode identifiers while data/video assets
    remain task-only.
    """

    plan: dict[str, str] = {}

    required_metadata = (
        "meta/tasks.parquet",
        "meta/tasks.jsonl",
    )
    for raw_path in required_metadata:
        plan[raw_path] = "metadata"
    episode_metadata = sorted((source_root / "meta/episodes").glob("*/*.parquet"))
    if not episode_metadata:
        raise ConfigurationError(
            f"no LeRobot v3 episode metadata found under {source_root / 'meta/episodes'}"
        )
    for path in episode_metadata:
        plan[str(path.relative_to(source_root))] = "metadata"

    for optional_path in ("README.md", "LICENSE"):
        if (source_root / optional_path).is_file():
            plan[optional_path] = "auxiliary"

    for episode in episodes:
        data_path = _safe_relative_path(
            episode.data.relative_path,
            field=f"episode {episode.episode_index} data.relative_path",
        )
        plan[str(data_path)] = "data"
        if set(episode.videos) != set(RGB_VIDEO_KEYS):
            raise SchemaValidationError(
                f"episode {episode.episode_index} must reference exactly the three RGB keys; "
                f"got {sorted(episode.videos)}"
            )
        for video_key in RGB_VIDEO_KEYS:
            video_path = _safe_relative_path(
                episode.videos[video_key].relative_path,
                field=f"episode {episode.episode_index} videos[{video_key!r}].relative_path",
            )
            if video_path.parts[0] != "videos" or video_key not in video_path.parts:
                raise SchemaValidationError(
                    f"episode {episode.episode_index} has inconsistent RGB path for {video_key}: "
                    f"{video_path}"
                )
            plan[str(video_path)] = "rgb_video"
        if episode.annotation_path:
            annotation_path = _safe_relative_path(
                episode.annotation_path,
                field=f"episode {episode.episode_index} annotation_path",
            )
            plan[str(annotation_path)] = "annotation"

    missing = sorted(relative for relative in plan if not (source_root / relative).is_file())
    if missing:
        raise ConfigurationError(
            f"{len(missing)} explicitly referenced source files are missing; "
            f"first entries: {missing[:20]}"
        )
    resolved_root = source_root.resolve()
    escaped: list[str] = []
    for relative in plan:
        resolved_source = (source_root / relative).resolve()
        try:
            resolved_source.relative_to(resolved_root)
        except ValueError:
            escaped.append(relative)
    if escaped:
        raise SchemaValidationError(
            "explicitly referenced source paths must not escape the pinned dataset root "
            f"through symlinks; first entries: {escaped[:20]}"
        )
    return plan


def materialize_behavior_view(
    *,
    view_dir: str | Path,
    output_root: str | Path,
    mode: MaterializationMode = "hardlink",
    source_root_override: str | Path | None = None,
) -> dict[str, object]:
    """Project one virtual task view into a GPU-visible LeRobotDataset v3 root."""

    if mode not in {"hardlink", "copy"}:
        raise ConfigurationError(f"unsupported materialization mode: {mode!r}")
    resolved_view = Path(view_dir).expanduser().resolve()
    manifest = _read_json_object(resolved_view / "view_manifest.json")
    manifest_video_keys = tuple(str(key) for key in manifest.get("video_keys", ()))
    if manifest_video_keys != RGB_VIDEO_KEYS:
        raise SchemaValidationError(
            "materialization requires exactly the canonical three RGB keys and no depth; "
            f"got {manifest_video_keys}"
        )
    episodes_name = str(manifest.get("episodes_file", "episodes.jsonl"))
    episodes_relative = _safe_relative_path(episodes_name, field="episodes_file")
    episodes = _load_episodes(resolved_view / episodes_relative)
    expected_count = manifest.get("episode_count")
    if expected_count is not None and int(expected_count) != len(episodes):
        raise SchemaValidationError(
            f"view declares {expected_count} episodes but {len(episodes)} were loaded"
        )

    source_revision = str(manifest.get("source_revision", "")).strip()
    if not source_revision:
        raise SchemaValidationError("view_manifest.json does not define source_revision")
    source_root = _resolve_source_root(manifest, source_root_override)
    _validate_source_revision(source_root, source_revision)
    plan = _build_source_plan(source_root=source_root, episodes=episodes)

    destination_root = Path(output_root).expanduser().resolve()
    _prepare_output_parent(destination_root)
    if mode == "hardlink" and source_root.stat().st_dev != destination_root.parent.stat().st_dev:
        raise ConfigurationError(
            "hardlink materialization requires source and output on the same filesystem: "
            f"source={source_root}, output={destination_root}. No copy fallback was attempted; "
            "use --mode copy explicitly if the additional storage use is acceptable."
        )
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_root.name}.materializing-",
            dir=destination_root.parent,
        )
    )
    try:
        category_counts: dict[str, int] = defaultdict(int)
        category_bytes: dict[str, int] = defaultdict(int)
        category_reused: dict[str, int] = defaultdict(int)
        total_bytes = 0
        inode_reuse_count = 0
        for relative, category in sorted(plan.items()):
            size, inode_reused = _copy_or_link(
                source_root / relative,
                staging_root / relative,
                mode=mode,
            )
            category_counts[category] += 1
            category_bytes[category] += size
            category_reused[category] += int(inode_reused)
            total_bytes += size
            inode_reuse_count += int(inode_reused)

        # info/stats must be rewritten instead of linked: depth is deliberately
        # absent from both the feature contract and the projected physical assets.
        _narrow_info(source_root / "meta/info.json", staging_root / "meta/info.json")
        _narrow_stats(source_root / "meta/stats.json", staging_root / "meta/stats.json")
        (staging_root / ".dataset_revision").write_text(
            source_revision + "\n",
            encoding="utf-8",
        )

        generated_paths = (
            staging_root / "meta/info.json",
            staging_root / "meta/stats.json",
            staging_root / ".dataset_revision",
        )
        generated_bytes = sum(path.stat().st_size for path in generated_paths)
        inventory = {
            category: {
                "file_count": category_counts[category],
                "bytes": category_bytes[category],
                "inode_reuse_file_count": category_reused[category],
            }
            for category in sorted(category_counts)
        }
        result: dict[str, object] = {
            "schema_version": "1.0",
            "source": {
                "repo_id": str(manifest.get("source_repo_id", "")),
                "revision": source_revision,
                "root": str(source_root),
                "view_manifest": str((resolved_view / "view_manifest.json").resolve()),
                "episodes_manifest": str((resolved_view / episodes_relative).resolve()),
            },
            "projection": {
                "mode": mode,
                "output_root": str(destination_root),
                "task": manifest.get("task"),
                "episode_count": len(episodes),
                "video_keys": list(RGB_VIDEO_KEYS),
                "includes_depth": False,
            },
            "inventory": {
                "source_file_count": len(plan),
                "source_bytes": total_bytes,
                "generated_file_count": len(generated_paths),
                "generated_bytes": generated_bytes,
                "materialized_file_count": len(plan) + len(generated_paths),
                "materialized_bytes": total_bytes + generated_bytes,
                "inode_reuse_file_count": inode_reuse_count,
                "inode_reuse_bytes": (
                    sum(
                        category_bytes[category]
                        for category in category_bytes
                        if category_reused[category] == category_counts[category]
                    )
                    if mode == "hardlink"
                    else 0
                ),
                "categories": inventory,
                "manifest_file_excluded_from_totals": True,
            },
        }
        _atomic_write_json(staging_root / MATERIALIZATION_MANIFEST, result)
        if destination_root.exists():
            raise ConfigurationError(
                f"materialization target appeared during build: {destination_root}"
            )
        os.rename(staging_root, destination_root)
        return result
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
