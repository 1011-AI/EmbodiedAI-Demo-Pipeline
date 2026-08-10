from __future__ import annotations

import json
from pathlib import Path

from embodied_demo.behavior1k.dataset import (
    _read_json,
    load_episode_references,
)
from embodied_demo.behavior1k.schemas import (
    BehaviorDatasetConfig,
    BehaviorStatsReference,
    BehaviorTaskConfig,
    BehaviorViewManifest,
    EpisodeReference,
)
from embodied_demo.behavior1k.stats import (
    ComputedBehaviorStats,
    compute_selected_episode_stats,
)
from embodied_demo.errors import SchemaValidationError


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _select_task_episodes(
    episodes: list[EpisodeReference],
    task: BehaviorTaskConfig,
) -> list[EpisodeReference]:
    if task.episode_indices is None:
        return episodes
    requested = set(task.episode_indices)
    selected = [item for item in episodes if item.episode_index in requested]
    found = {item.episode_index for item in selected}
    missing = sorted(requested - found)
    if missing:
        raise SchemaValidationError(
            f"task view requested episode indices not present in metadata: {missing[:20]}"
        )
    return selected


def write_virtual_view(
    *,
    output_dir: Path,
    source_root: Path,
    config: BehaviorDatasetConfig,
    task: BehaviorTaskConfig,
    episodes: list[EpisodeReference],
    stats: ComputedBehaviorStats | None = None,
) -> BehaviorViewManifest:
    selected = _select_task_episodes(episodes, task)

    frame_count = sum(item.length for item in selected)
    if stats is not None and stats.frame_count != frame_count:
        raise SchemaValidationError(
            "computed statistics frame count does not match the selected view: "
            f"stats={stats.frame_count}, selected={frame_count}"
        )
    stats_reference = (
        BehaviorStatsReference(
            frame_count=stats.frame_count,
            quantile_method=stats.quantile_method,
            quantile_sample_count=stats.quantile_sample_count,
            quantile_sample_limit=stats.quantile_sample_limit,
        )
        if stats is not None
        else None
    )
    manifest = BehaviorViewManifest(
        source_repo_id=config.dataset.repo_id,
        source_revision=config.dataset.revision,
        source_root=str(source_root),
        task=task,
        video_keys=config.selection.video_keys,
        episode_count=len(selected),
        frame_count=frame_count,
        stats=stats_reference,
    )
    episodes_content = "".join(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        for item in selected
    )
    _atomic_write(output_dir / manifest.episodes_file, episodes_content)
    if stats is not None and manifest.stats is not None:
        _atomic_write(
            output_dir / manifest.stats.policy_file,
            json.dumps(stats.policy_stats, ensure_ascii=False, indent=2) + "\n",
        )
        _atomic_write(
            output_dir / manifest.stats.raw_state_audit_file,
            json.dumps(stats.raw_state_stats, ensure_ascii=False, indent=2) + "\n",
        )
    _atomic_write(
        output_dir / "view_manifest.json",
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def prepare_virtual_view(
    *,
    source_root: Path,
    output_dir: Path,
    config: BehaviorDatasetConfig,
    task: BehaviorTaskConfig,
) -> BehaviorViewManifest:
    info = _read_json(source_root / "meta/info.json")
    episodes = load_episode_references(
        source_root,
        task_index=task.task_index,
        video_keys=config.selection.video_keys,
        info=info,
    )
    selected = _select_task_episodes(episodes, task)
    stats = compute_selected_episode_stats(
        source_root=source_root,
        episodes=selected,
    )
    return write_virtual_view(
        output_dir=output_dir,
        source_root=source_root,
        config=config,
        task=task,
        episodes=selected,
        stats=stats,
    )
