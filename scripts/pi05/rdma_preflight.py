#!/usr/bin/env python3
from __future__ import annotations

"""Fail-fast host RDMA inspection for the π0.5 Comet JAX runner.

This checks that the host can expose an IB/RoCE transport to NCCL.  It does not
claim that GPUDirect RDMA is active: that is only established by the collective
probe and its NCCL log, which run immediately after this script on multi-node
jobs.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import socket
import tempfile
from typing import Any, Callable


RDMA_LIBRARIES = ("libibverbs.so.1", "libmlx5.so.1", "librdmacm.so.1")
FALSE_VALUES = {"", "0", "false", "no", "off"}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def inspect_rdma(
    *,
    mode: str,
    device_root: Path = Path("/dev/infiniband"),
    sys_class_root: Path = Path("/sys/class/infiniband"),
    module_root: Path = Path("/sys/module"),
    library_loader: Callable[[str], Any] = ctypes.CDLL,
) -> tuple[dict[str, Any], list[str]]:
    if mode not in {"auto", "required", "disabled"}:
        raise ValueError(f"unsupported RDMA mode: {mode}")

    devices = sorted(path.name for path in device_root.iterdir()) if device_root.is_dir() else []
    missing_libraries: list[str] = []
    for library in RDMA_LIBRARIES:
        try:
            library_loader(library)
        except OSError:
            missing_libraries.append(library)

    ports: list[dict[str, Any]] = []
    if sys_class_root.is_dir():
        for hca in sorted(path for path in sys_class_root.iterdir() if path.is_dir()):
            ports_root = hca / "ports"
            if not ports_root.is_dir():
                continue
            for port in sorted(path for path in ports_root.iterdir() if path.is_dir()):
                state = _read(port / "state")
                link_layer = _read(port / "link_layer")
                gid_types_root = port / "gid_attrs/types"
                gid_types = (
                    sorted({_read(path) for path in gid_types_root.iterdir()} - {""})
                    if gid_types_root.is_dir()
                    else []
                )
                ports.append(
                    {
                        "hca": hca.name,
                        "port": port.name,
                        "state": state,
                        "active": state.upper().startswith("4:") or "ACTIVE" in state.upper(),
                        "link_layer": link_layer,
                        "gid_types": gid_types,
                    }
                )

    active_ports = [port for port in ports if port["active"]]
    active_roce = [
        port
        for port in active_ports
        if str(port["link_layer"]).lower() == "ethernet"
        and any("roce" in value.lower() for value in port["gid_types"])
    ]
    active_native_ib = [
        port for port in active_ports if str(port["link_layer"]).lower() == "infiniband"
    ]
    peermem_loaded = (module_root / "nvidia_peermem").is_dir()
    dmabuf_allowed = os.environ.get("NCCL_DMABUF_ENABLE", "1").strip().lower() not in FALSE_VALUES

    errors: list[str] = []
    nccl_net = os.environ.get("NCCL_NET", "IB").strip()
    ib_disabled = os.environ.get("NCCL_IB_DISABLE", "0").strip().lower()
    if mode != "disabled" and nccl_net.upper() != "IB":
        errors.append(f"NCCL_NET must be IB, got {nccl_net!r}")
    if mode != "disabled" and ib_disabled not in FALSE_VALUES:
        errors.append(f"NCCL_IB_DISABLE must be 0, got {ib_disabled!r}")
    if missing_libraries:
        errors.append("missing RDMA libraries: " + ", ".join(missing_libraries))
    if "rdma_cm" not in devices or not any(name.startswith("uverbs") for name in devices):
        errors.append("missing /dev/infiniband/rdma_cm or uverbs* device")
    if not active_ports:
        errors.append("no active /sys/class/infiniband HCA port")

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "host": socket.gethostname(),
        "mode": mode,
        "nccl_net": nccl_net,
        "nccl_ib_disable": os.environ.get("NCCL_IB_DISABLE", "0"),
        "devices": devices,
        "missing_libraries": missing_libraries,
        "ports": ports,
        "active_transport": (
            "roce" if active_roce else "native_ib" if active_native_ib else "unknown"
        ),
        "roce_ready": bool(active_roce),
        "native_ib_ready": bool(active_native_ib),
        "gdr_preflight": {
            "nvidia_peermem_loaded": peermem_loaded,
            "nccl_dmabuf_allowed": dmabuf_allowed,
            "status": (
                "peermem_available"
                if peermem_loaded
                else "dmabuf_allowed_unverified"
                if dmabuf_allowed
                else "no_known_gdr_path"
            ),
            "actual_status": "requires_nccl_collective_log",
        },
        "ok": not errors,
        "errors": errors,
    }
    if mode == "disabled":
        report["ok"] = True
    elif mode == "auto" and errors:
        report["ok"] = False
    return report, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("auto", "required", "disabled"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-root", type=Path, default=Path("/dev/infiniband"))
    parser.add_argument("--sys-class-root", type=Path, default=Path("/sys/class/infiniband"))
    args = parser.parse_args(argv)
    report, errors = inspect_rdma(
        mode=args.mode,
        device_root=args.device_root,
        sys_class_root=args.sys_class_root,
    )
    _atomic_json(args.output, report)
    print("PI05_RDMA_PREFLIGHT " + json.dumps(report, sort_keys=True), flush=True)
    if args.mode == "required" and errors:
        raise SystemExit("ERROR: RDMA preflight failed; refusing Socket fallback: " + "; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
