#!/usr/bin/env python3
from __future__ import annotations

"""Image/startup self-check for Demo Pipeline π0.5 Comet continuation."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

PROJECT_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE_ROOT))

from embodied_demo.pi05_backend_integrity import (
    BackendIntegrityError,
    verify_prepared_backends,
)


EXPECTED = {
    "jax": "0.5.3",
    "jaxlib": "0.5.3",
    "jax-cuda12-plugin": "0.5.3",
    "jax-cuda12-pjrt": "0.5.3",
    "flax": "0.10.2",
    "orbax-checkpoint": "0.11.13",
    "numpy": "1.26.4",
    "transformers": "4.53.2",
    "lerobot": "0.5.2",
}
COMET_COMMIT = "4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5"
LEROBOT_COMMIT = "c43f58116b975ae79af62714e1417b38facd4e37"


def project_root() -> Path:
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "pipelines").is_dir():
            return candidate
    raise SystemExit("ERROR: cannot locate EmbodiedAI-Demo-Pipeline")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpus", type=int, default=1)
    args = parser.parse_args(argv)
    root = project_root()
    errors: list[str] = []
    versions: dict[str, str] = {}
    for package, expected in EXPECTED.items():
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
        if versions[package] != expected:
            errors.append(f"{package}: expected {expected}, got {versions[package]}")

    import jax

    gpus = [device for device in jax.devices() if device.platform == "gpu"]
    if len(gpus) < args.require_gpus:
        errors.append(f"JAX GPUs: require {args.require_gpus}, got {len(gpus)}")
    backend_integrity = None
    try:
        backend_integrity = verify_prepared_backends(root)
    except BackendIntegrityError as exc:
        errors.append(f"prepared backends: {exc}")
    prepared = backend_integrity["backends"] if backend_integrity else {}
    commit = str(prepared.get("openpi_comet", {}).get("revision", "unverified"))
    if commit != COMET_COMMIT:
        errors.append(f"OpenPI-Comet commit: expected {COMET_COMMIT}, got {commit}")
    lerobot_commit = str(prepared.get("lerobot", {}).get("revision", "unverified"))
    if lerobot_commit != LEROBOT_COMMIT:
        errors.append(f"LeRobot commit: expected {LEROBOT_COMMIT}, got {lerobot_commit}")
    checkpoint = root / "models/openpi_comet/pi05-b1kpt50-cs32"
    verify = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/pi05/verify_comet_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    checkpoint_report = None
    try:
        checkpoint_report = json.loads(verify.stdout)
    except json.JSONDecodeError:
        errors.append("checkpoint verification did not return JSON")
    if verify.returncode:
        errors.append("official checkpoint is incomplete")
    contracts = root / "data/custom/pi05_comet/behavior1k"
    required_contracts = (
        "all_tasks_train19800_seed42_h32_sampling.json",
        "all_tasks_val200_seed42_h32_sampling.json",
        "dataset_fingerprint.json",
        "behavior1k_all_contract.json",
        "normalization_audit.json",
    )
    missing_contracts = [name for name in required_contracts if not (contracts / name).is_file()]
    if missing_contracts:
        errors.append("missing data contracts: " + ", ".join(missing_contracts))
    payload = {
        "schema_version": "1.0",
        "python": sys.version.split()[0],
        "packages": versions,
        "jax_gpu_count": len(gpus),
        "jax_devices": [str(device) for device in gpus],
        "openpi_comet_commit": commit,
        "lerobot_commit": lerobot_commit,
        "prepared_backend_integrity": backend_integrity,
        "checkpoint": checkpoint_report,
        "missing_data_contracts": missing_contracts,
        "ok": not errors,
        "errors": errors,
    }
    print("PI05_COMET_DOCTOR " + json.dumps(payload, sort_keys=True), flush=True)
    if errors:
        raise SystemExit("ERROR: " + "; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
