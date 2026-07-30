from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from pipelines.custom.fastwam.behavior1k.prepare import discover_task_selection
from pipelines.custom.fastwam.behavior1k.v3_shards import (
    filter_v3_hf_dataset,
    load_v3_episode_metadata,
    read_v3_episode_table,
    shift_v3_video_timestamps,
    v3_data_file_path,
    v3_video_file_path,
)


VIDEO_KEY = "observation.rgb.zed_link_camera_0"


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
