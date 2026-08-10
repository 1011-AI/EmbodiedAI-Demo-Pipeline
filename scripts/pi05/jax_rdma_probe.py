#!/usr/bin/env python3
from __future__ import annotations

"""Run a real multi-host JAX collective and prove its NCCL data transport."""

import argparse
import functools
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any


IB_PATTERN = re.compile(r"(?:NET/IB|network\s+IB)", re.IGNORECASE)
SOCKET_PATTERN = re.compile(r"NET/Socket", re.IGNORECASE)
GDR_PATTERN = re.compile(r"(?:/GDRDMA|GPUDirect\s+RDMA|GPU\s+Direct\s+RDMA)", re.IGNORECASE)


def parse_nccl_transport(text: str) -> dict[str, Any]:
    has_ib = bool(IB_PATTERN.search(text))
    has_socket = bool(SOCKET_PATTERN.search(text))
    gdr = bool(GDR_PATTERN.search(text))
    transport = (
        "mixed_ib_socket"
        if has_ib and has_socket
        else "ib"
        if has_ib
        else "socket"
        if has_socket
        else "unknown"
    )
    return {
        "transport": transport,
        "net_ib_observed": has_ib,
        "net_socket_observed": has_socket,
        "gdr_observed": gdr,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _read_logs(pattern: str) -> tuple[list[str], str]:
    paths = sorted(Path().glob(pattern) if not Path(pattern).is_absolute() else Path(pattern).parent.glob(Path(pattern).name))
    chunks: list[str] = []
    for path in paths:
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return [str(path) for path in paths], "\n".join(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--local-device-count", type=int, required=True)
    parser.add_argument("--log-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elements-per-device", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args(argv)

    if args.num_processes <= 1:
        raise SystemExit("ERROR: JAX RDMA probe requires at least two nodes")
    import jax
    import jax.numpy as jnp

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
        local_device_ids=list(range(args.local_device_count)),
    )
    try:
        if jax.local_device_count() != args.local_device_count:
            raise RuntimeError(
                f"local device mismatch: expected {args.local_device_count}, "
                f"JAX sees {jax.local_device_count()}"
            )

        @functools.partial(jax.pmap, axis_name="global_devices")
        def all_reduce(value):
            return jax.lax.psum(value, "global_devices")

        value = jnp.ones(
            (args.local_device_count, args.elements_per_device), dtype=jnp.float32
        )
        result = all_reduce(value)
        jax.block_until_ready(result)
        elapsed: list[float] = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            result = all_reduce(value)
            jax.block_until_ready(result)
            elapsed.append(time.perf_counter() - start)
        expected = float(jax.device_count())
        observed = float(result[0, 0])
        if observed != expected:
            raise RuntimeError(f"all-reduce value mismatch: expected {expected}, got {observed}")
        topology = {
            "process_count": jax.process_count(),
            "process_index": jax.process_index(),
            "local_device_count": jax.local_device_count(),
            "global_device_count": jax.device_count(),
        }
    finally:
        jax.distributed.shutdown()

    # NCCL_DEBUG_FILE is flushed as communicators are destroyed/shutdown.
    time.sleep(0.5)
    log_paths, log_text = _read_logs(args.log_glob)
    transport = parse_nccl_transport(log_text)
    payload = {
        "schema_version": "1.0",
        **topology,
        "collective": {
            "elements_per_device": args.elements_per_device,
            "bytes_per_device": args.elements_per_device * 4,
            "iterations": args.iterations,
            "elapsed_seconds": elapsed,
            "mean_seconds": sum(elapsed) / len(elapsed),
            "value": observed,
        },
        "nccl_logs": log_paths,
        **transport,
        "ok": transport["net_ib_observed"] and not transport["net_socket_observed"],
    }
    _atomic_json(args.output, payload)
    print("PI05_JAX_RDMA_PROBE " + json.dumps(payload, sort_keys=True), flush=True)
    if not log_paths:
        raise SystemExit("ERROR: NCCL emitted no probe log; cannot prove RDMA transport")
    if transport["net_socket_observed"]:
        raise SystemExit("ERROR: NCCL data transport contains NET/Socket; refusing training")
    if not transport["net_ib_observed"]:
        raise SystemExit("ERROR: NCCL log contains no NET/IB data transport; refusing training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
