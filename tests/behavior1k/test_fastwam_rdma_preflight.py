from __future__ import annotations

import os
from pathlib import Path

import pytest

from experiments.custom.fastwam_behavior1k_all.run import (
    RDMA_RUNTIME_LIBRARIES,
    require_multinode_rdma_runtime,
)


def _touch_devices(root: Path) -> None:
    (root / "rdma_cm").touch()
    (root / "uverbs2").touch()


def test_multinode_rdma_preflight_forces_ib(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _touch_devices(tmp_path)
    loaded = []
    monkeypatch.delenv("NCCL_NET", raising=False)
    monkeypatch.setenv("NCCL_IB_DISABLE", "0")

    require_multinode_rdma_runtime(
        require_devices=True,
        device_root=tmp_path,
        library_loader=lambda name: loaded.append(name),
    )

    assert loaded == list(RDMA_RUNTIME_LIBRARIES)
    assert os.environ["NCCL_NET"] == "IB"


def test_multinode_rdma_preflight_rejects_missing_userspace_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _touch_devices(tmp_path)
    monkeypatch.setenv("NCCL_NET", "IB")
    monkeypatch.setenv("NCCL_IB_DISABLE", "0")

    def load_library(name: str) -> None:
        if name == "libibverbs.so.1":
            raise OSError("missing")

    with pytest.raises(SystemExit, match="libibverbs.so.1"):
        require_multinode_rdma_runtime(
            require_devices=True,
            device_root=tmp_path,
            library_loader=load_library,
        )


def test_multinode_rdma_preflight_rejects_missing_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NCCL_NET", "IB")
    monkeypatch.setenv("NCCL_IB_DISABLE", "0")

    with pytest.raises(SystemExit, match="/dev/infiniband"):
        require_multinode_rdma_runtime(
            require_devices=True,
            device_root=tmp_path,
            library_loader=lambda _name: None,
        )


def test_multinode_rdma_preflight_dry_run_skips_device_requirement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NCCL_NET", raising=False)
    monkeypatch.delenv("NCCL_IB_DISABLE", raising=False)

    require_multinode_rdma_runtime(
        require_devices=False,
        device_root=tmp_path,
        library_loader=lambda _name: None,
    )

    assert os.environ["NCCL_NET"] == "IB"
