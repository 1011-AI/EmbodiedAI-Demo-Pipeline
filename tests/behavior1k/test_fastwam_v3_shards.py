from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from experiments.custom.fastwam_behavior1k_task0.run import (
    _dry_run_dataset_placeholder,
    _expected_text_embedding_path,
    _text_embedding_command,
)
from pipelines.custom.fastwam.behavior1k.prepare import discover_task_selection
from pipelines.custom.fastwam.behavior1k.prepare import (
    TaskEpisodeSelection,
    build_dataset_fingerprint,
    partition_episode_indices,
    validate_task_norm_stats,
)
from pipelines.custom.fastwam.behavior1k.v3_shards import (
    filter_v3_hf_dataset,
    LazyV3ParquetDataset,
    load_v3_episode_metadata,
    read_v3_episode_table,
    shift_v3_video_timestamps,
    v3_data_file_path,
    v3_video_file_path,
)


VIDEO_KEY = "observation.rgb.zed_link_camera_0"


def test_dry_run_dataset_placeholder_is_not_an_omegaconf_interpolation(
    tmp_path: Path,
) -> None:
    placeholder = _dry_run_dataset_placeholder(tmp_path, "BEHAVIOR1K_DATA_ROOT")

    assert placeholder == str(
        tmp_path / "data/behavior1k/UNSET_BEHAVIOR1K_DATA_ROOT"
    )
    assert "${" not in placeholder


def test_task0_text_embedding_cache_name_matches_fastwam_precompute() -> None:
    path = _expected_text_embedding_path(
        Path("/cache"),
        task_instruction=(
            "Turn on the radio receiver that's on the table in the living room."
        ),
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        context_len=128,
    )

    assert path.name == (
        "3c6c057f0dd8a659b46826353422f560eb1a024e00c6cc9fc5f814ee10a84338."
        "t5_len128.wan22ti2v5b.pt"
    )


def test_text_embedding_command_uses_generated_task_config() -> None:
    command = _text_embedding_command(
        source_root=Path("/fastwam"),
        task_name="behavior1k_task0_action_only",
        fastwam_config={
            "model_id": "Wan-AI/Wan2.2-TI2V-5B",
            "tokenizer_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
            "redirect_common_files": False,
        },
        overwrite=False,
    )

    assert command[1] == "/fastwam/scripts/precompute_text_embeds.py"
    assert "task=behavior1k_task0_action_only" in command
    assert "model.redirect_common_files=false" in command
    assert "+overwrite=false" in command


def test_task0_entry_exposes_stable_runtime_arguments() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "experiments/custom/fastwam_behavior1k_task0/run.py"
    ).read_text(encoding="utf-8")

    assert '"--dataset-root"' in source
    assert '"--profile"' in source
    assert '"--run-id"' in source
    assert '"--checkpoint-mode"' in source
    assert '"--continuation-mode"' in source
    assert '"--weights-checkpoint"' in source
    assert '"--resume-state"' in source
    assert "args.dataset_root" in source
    assert 'os.environ["FASTWAM_RUN_ID"] = args.run_id' in source
    assert 'command.extend(["--profile", args.profile])' in source


def _write_real_v3_structure(root: Path) -> list[dict[str, Any]]:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(exist_ok=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "R1Pro",
        "total_episodes": 3,
        "total_frames": 6,
        "total_tasks": 2,
        "chunks_size": 1000,
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/"
            "file-{file_index:03d}.mp4"
        ),
        "features": {
            "action": {"dtype": "float32", "shape": [23]},
            "observation.state": {"dtype": "float32", "shape": [61]},
            VIDEO_KEY: {"dtype": "video", "shape": [3, 720, 720]},
        },
    }
    (root / "meta/info.json").write_text(
        json.dumps(info),
        encoding="utf-8",
    )
    (root / "meta/tasks.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_index": 0,
                        "task_name": "turning_on_radio",
                        "task": "Turn on the radio.",
                    }
                ),
                json.dumps(
                    {
                        "task_index": 1,
                        "task_name": "other",
                        "task": "Do another task.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "meta/stats.json").write_text(
        json.dumps({"action": {"mean": [0.0] * 23}}),
        encoding="utf-8",
    )
    rows = [
        {
            "episode_index": 0,
            "tasks": ["turning_on_radio"],
            "task_index": 0,
            "length": 2,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 0,
            "dataset_to_index": 2,
            f"videos/{VIDEO_KEY}/chunk_index": 0,
            f"videos/{VIDEO_KEY}/file_index": 0,
            f"videos/{VIDEO_KEY}/from_timestamp": 0.0,
            f"videos/{VIDEO_KEY}/to_timestamp": 2 / 30,
        },
        {
            "episode_index": 1,
            "tasks": ["other"],
            "task_index": 1,
            "length": 3,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 2,
            "dataset_to_index": 5,
            f"videos/{VIDEO_KEY}/chunk_index": 0,
            f"videos/{VIDEO_KEY}/file_index": 0,
            f"videos/{VIDEO_KEY}/from_timestamp": 2 / 30,
            f"videos/{VIDEO_KEY}/to_timestamp": 5 / 30,
        },
        {
            "episode_index": 2,
            "tasks": ["turning_on_radio"],
            "task_index": 0,
            "length": 1,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 5,
            "dataset_to_index": 6,
            f"videos/{VIDEO_KEY}/chunk_index": 0,
            f"videos/{VIDEO_KEY}/file_index": 0,
            f"videos/{VIDEO_KEY}/from_timestamp": 5 / 30,
            f"videos/{VIDEO_KEY}/to_timestamp": 6 / 30,
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(rows),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    return rows


class _FakeHFDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def filter(
        self,
        function,
        *,
        input_columns: list[str],
        batched: bool,
        desc: str,
    ) -> "_FakeHFDataset":
        assert input_columns == ["episode_index"]
        assert batched is True
        assert desc
        mask = function([row["episode_index"] for row in self.rows])
        return _FakeHFDataset(
            [row for row, keep in zip(self.rows, mask, strict=True) if keep]
        )

    def __getitem__(self, key: str):
        return [row[key] for row in self.rows]


def test_real_v3_episode_structure_uses_flattened_refs_and_unique_shards(
    tmp_path: Path,
) -> None:
    rows = _write_real_v3_structure(tmp_path)

    metadata = load_v3_episode_metadata(tmp_path)
    selection = discover_task_selection(
        tmp_path,
        task_index=0,
        expected_task_name="turning_on_radio",
    )

    assert list(metadata) == [0, 1, 2]
    assert metadata[1]["data/file_index"] == 0
    assert selection.episode_indices == (0, 2)
    assert selection.task_instruction == "Turn on the radio."
    assert selection.data_shards == ("data/chunk-000/file-000.parquet",)
    assert v3_data_file_path(
        json.loads((tmp_path / "meta/info.json").read_text()),
        rows[2],
    ) == Path("data/chunk-000/file-000.parquet")
    assert v3_video_file_path(
        json.loads((tmp_path / "meta/info.json").read_text()),
        rows[2],
        VIDEO_KEY,
    ) == Path(f"videos/{VIDEO_KEY}/chunk-000/file-000.mp4")


def test_dataset_fingerprint_binds_metadata_and_episode_selection(
    tmp_path: Path,
) -> None:
    _write_real_v3_structure(tmp_path)
    selection = discover_task_selection(tmp_path)

    first = build_dataset_fingerprint(tmp_path, selection)
    second = build_dataset_fingerprint(tmp_path, selection)

    assert first == second
    assert len(first["sha256"]) == 64
    assert len(first["selection_sha256"]) == 64
    assert {item["path"] for item in first["metadata_files"]} == {
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/stats.json",
        "meta/episodes/chunk-000/file-000.parquet",
    }

    (tmp_path / "meta/stats.json").write_text(
        json.dumps({"action": {"mean": [1.0] * 23}}),
        encoding="utf-8",
    )
    changed = build_dataset_fingerprint(tmp_path, selection)
    assert changed["sha256"] != first["sha256"]
    assert changed["selection_sha256"] == first["selection_sha256"]


def test_episode_partition_is_deterministic_disjoint_and_seeded() -> None:
    first = partition_episode_indices(
        tuple(range(200)),
        validation_proportion=0.01,
        seed=42,
    )
    same = partition_episode_indices(
        tuple(reversed(range(200))),
        validation_proportion=0.01,
        seed=42,
    )
    changed = partition_episode_indices(
        tuple(range(200)),
        validation_proportion=0.01,
        seed=43,
    )

    assert first == same
    assert len(first.train_episode_indices) == 198
    assert len(first.val_episode_indices) == 2
    assert not set(first.train_episode_indices) & set(first.val_episode_indices)
    assert first.sha256 == same.sha256
    assert first.val_episode_indices != changed.val_episode_indices


def test_norm_stats_validation_binds_exact_training_split(tmp_path: Path) -> None:
    selection = TaskEpisodeSelection(
        task_index=0,
        task_name="turning_on_radio",
        task_instruction="Turn on the radio.",
        episode_indices=(0, 2),
        data_shards=("data/chunk-000/file-000.parquet",),
    )
    vector = [0.0] * 23
    payload = {
        "state": {
            "default": {
                "global_mean": vector,
                "global_std": [1.0] * 23,
                "global_min": vector,
                "global_max": vector,
            }
        },
        "action": {
            "default": {
                "global_mean": vector,
                "global_std": [1.0] * 23,
                "global_min": vector,
                "global_max": vector,
            }
        },
        "num_episodes": 2,
        "provenance": {
            "episode_indices": [0, 2],
            "action_semantics": "raw mixed 23D; no global delta transform",
        },
    }
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_task_norm_stats(path, selection) == payload
    payload["provenance"]["episode_indices"] = [0, 1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="episode ids"):
        validate_task_norm_stats(path, selection)


def test_shared_v3_parquet_is_filtered_to_exact_selected_episode_rows(
    tmp_path: Path,
) -> None:
    metadata_rows = _write_real_v3_structure(tmp_path)
    metadata = {int(row["episode_index"]): row for row in metadata_rows}
    shared_rows = [
        {"episode_index": 0, "value": 0},
        {"episode_index": 0, "value": 1},
        {"episode_index": 1, "value": 2},
        {"episode_index": 1, "value": 3},
        {"episode_index": 1, "value": 4},
        {"episode_index": 2, "value": 5},
    ]

    filtered = filter_v3_hf_dataset(
        _FakeHFDataset(shared_rows),
        [0, 2],
        metadata,
    )

    assert filtered["episode_index"] == [0, 0, 2]
    assert filtered["value"] == [0, 1, 5]
    with pytest.raises(ValueError, match="unique"):
        filter_v3_hf_dataset(_FakeHFDataset(shared_rows), [0, 0], metadata)


def test_direct_episode_read_and_video_timestamp_shift_do_not_leak_neighbors(
    tmp_path: Path,
) -> None:
    metadata_rows = _write_real_v3_structure(tmp_path)
    data_path = tmp_path / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0, 1, 1, 1, 2],
                "value": [0, 1, 2, 3, 4, 5],
            }
        ),
        data_path,
    )

    table = read_v3_episode_table(data_path, 1, metadata_rows[1])
    shifted = shift_v3_video_timestamps(
        metadata_rows[1],
        VIDEO_KEY,
        [0.0, 1 / 30, 2 / 30],
    )

    assert table["episode_index"].to_pylist() == [1, 1, 1]
    assert table["value"].to_pylist() == [2, 3, 4]
    assert shifted == pytest.approx([2 / 30, 3 / 30, 4 / 30])


def test_lazy_v3_dataset_maps_selected_local_indices_and_reuses_shared_shard(
    tmp_path: Path,
) -> None:
    metadata_rows = _write_real_v3_structure(tmp_path)
    metadata = {int(row["episode_index"]): row for row in metadata_rows}
    data_path = tmp_path / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0, 1, 1, 1, 2],
                "task_index": [0, 0, 1, 1, 1, 0],
                "timestamp": [0.0, 1 / 30, 0.0, 1 / 30, 2 / 30, 0.0],
                "value": [0, 1, 2, 3, 4, 5],
            }
        ),
        data_path,
    )
    info = json.loads((tmp_path / "meta/info.json").read_text())
    info["features"] = {
        "episode_index": {"dtype": "int64"},
        "task_index": {"dtype": "int64"},
        "timestamp": {"dtype": "float32"},
        "value": {"dtype": "int64"},
        VIDEO_KEY: {"dtype": "video"},
    }
    dataset = LazyV3ParquetDataset(
        root=tmp_path,
        info=info,
        selected_episodes=[2, 0],
        episode_metadata=metadata,
        cache_size=1,
    )

    assert len(dataset) == 3
    assert dataset[0]["episode_index"].item() == 2
    assert dataset[0]["value"].item() == 5
    batch = dataset.select([1, 2])["value"]
    assert [value.item() for value in batch] == [0, 1]
    assert dataset.cache_misses == 1
    assert dataset.cache_hits > 0


def test_lazy_v3_dataset_rejects_neighbor_row_mapping(tmp_path: Path) -> None:
    metadata_rows = _write_real_v3_structure(tmp_path)
    metadata = {int(row["episode_index"]): row for row in metadata_rows}
    data_path = tmp_path / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    # Same row count, deliberately wrong episode layout.
    pq.write_table(
        pa.table(
            {
                "episode_index": [1, 0, 1, 1, 1, 2],
                "task_index": [0, 0, 1, 1, 1, 0],
                "timestamp": [0.0] * 6,
            }
        ),
        data_path,
    )
    info = json.loads((tmp_path / "meta/info.json").read_text())
    info["features"] = {
        "episode_index": {"dtype": "int64"},
        "task_index": {"dtype": "int64"},
        "timestamp": {"dtype": "float32"},
        VIDEO_KEY: {"dtype": "video"},
    }
    dataset = LazyV3ParquetDataset(
        root=tmp_path,
        info=info,
        selected_episodes=[0],
        episode_metadata=metadata,
        cache_size=1,
    )

    with pytest.raises(ValueError, match="leaked episode rows"):
        dataset[0]
