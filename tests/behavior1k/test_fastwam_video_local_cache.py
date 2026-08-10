from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "upstreams/FastWAM-realrobot/src/fastwam/datasets/lerobot/lerobot/datasets/video_local_cache.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("fastwam_video_local_cache_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_cache_disabled_returns_original_value(
    tmp_path: Path, monkeypatch,
) -> None:
    module = _load_module()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.delenv("FASTWAM_VIDEO_LOCAL_CACHE_DIR", raising=False)

    assert module.localize_video_path(str(source)) == str(source)


def test_local_cache_copy_is_atomic_reused_and_source_sensitive(
    tmp_path: Path, monkeypatch,
) -> None:
    module = _load_module()
    source = tmp_path / "source.mp4"
    cache = tmp_path / "cache"
    source.write_bytes(b"first-version")
    monkeypatch.setenv("FASTWAM_VIDEO_LOCAL_CACHE_DIR", str(cache))
    monkeypatch.setenv("FASTWAM_VIDEO_LOCAL_CACHE_MAX_GIB", "1")

    first = Path(module.localize_video_path(source))
    second = Path(module.localize_video_path(source))
    assert first == second
    assert first.read_bytes() == source.read_bytes()
    assert not list(cache.rglob("*.partial"))

    source.write_bytes(b"second-version")
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    changed = Path(module.localize_video_path(source))

    assert changed != first
    assert changed.read_bytes() == b"second-version"


def test_local_cache_capacity_evicts_old_unprotected_file(
    tmp_path: Path, monkeypatch,
) -> None:
    module = _load_module()
    cache = tmp_path / "cache"
    monkeypatch.setenv("FASTWAM_VIDEO_LOCAL_CACHE_DIR", str(cache))
    monkeypatch.setenv("FASTWAM_VIDEO_LOCAL_CACHE_MAX_GIB", "0.000000001")
    source_a = tmp_path / "a.mp4"
    source_b = tmp_path / "b.mp4"
    source_a.write_bytes(b"a" * 8)
    source_b.write_bytes(b"b" * 8)

    cached_a = Path(module.localize_video_path(source_a))
    cached_b = Path(module.localize_video_path(source_b))

    assert not cached_a.exists()
    assert cached_b.read_bytes() == source_b.read_bytes()
