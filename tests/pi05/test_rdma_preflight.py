from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/pi05/rdma_preflight.py"
SPEC = importlib.util.spec_from_file_location("pi05_rdma_preflight", SCRIPT)
assert SPEC and SPEC.loader
rdma = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rdma)


def _fake_hca(tmp_path: Path, *, link_layer: str = "Ethernet") -> tuple[Path, Path]:
    devices = tmp_path / "dev"
    devices.mkdir()
    (devices / "rdma_cm").touch()
    (devices / "uverbs0").touch()
    sys_class = tmp_path / "sys"
    port = sys_class / "mlx5_0/ports/1"
    (port / "gid_attrs/types").mkdir(parents=True)
    (port / "state").write_text("4: ACTIVE\n", encoding="utf-8")
    (port / "link_layer").write_text(link_layer + "\n", encoding="utf-8")
    (port / "gid_attrs/types/0").write_text("RoCE v2\n", encoding="utf-8")
    return devices, sys_class


def test_required_roce_preflight_succeeds(monkeypatch, tmp_path: Path) -> None:
    devices, sys_class = _fake_hca(tmp_path)
    monkeypatch.setenv("NCCL_NET", "IB")
    monkeypatch.setenv("NCCL_IB_DISABLE", "0")
    report, errors = rdma.inspect_rdma(
        mode="required",
        device_root=devices,
        sys_class_root=sys_class,
        module_root=tmp_path / "modules",
        library_loader=lambda _name: object(),
    )
    assert errors == []
    assert report["ok"] is True
    assert report["active_transport"] == "roce"
    assert report["gdr_preflight"]["actual_status"] == "requires_nccl_collective_log"


def test_required_preflight_rejects_socket_fallback(monkeypatch, tmp_path: Path) -> None:
    devices, sys_class = _fake_hca(tmp_path)
    monkeypatch.setenv("NCCL_NET", "Socket")
    report, errors = rdma.inspect_rdma(
        mode="required",
        device_root=devices,
        sys_class_root=sys_class,
        module_root=tmp_path / "modules",
        library_loader=lambda _name: object(),
    )
    assert report["ok"] is False
    assert any("NCCL_NET" in error for error in errors)


def test_preflight_distinguishes_roce_from_gdr(monkeypatch, tmp_path: Path) -> None:
    devices, sys_class = _fake_hca(tmp_path)
    modules = tmp_path / "modules"
    (modules / "nvidia_peermem").mkdir(parents=True)
    monkeypatch.setenv("NCCL_NET", "IB")
    report, _ = rdma.inspect_rdma(
        mode="required",
        device_root=devices,
        sys_class_root=sys_class,
        module_root=modules,
        library_loader=lambda _name: object(),
    )
    assert report["roce_ready"] is True
    assert report["gdr_preflight"]["nvidia_peermem_loaded"] is True
    assert report["gdr_preflight"]["actual_status"] == "requires_nccl_collective_log"
