#!/usr/bin/env python3
"""Config-driven BEHAVIOR-1K evaluator server for a real LeRobot PI0.5 checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

# Allow direct execution immediately after clone, before editable installation.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_PROJECT_ROOT, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime dependency failure.
    raise SystemExit("ERROR: PyYAML is required to read the server config") from exc

from embodied_demo.behavior1k.r1pro import ACTION_DIM, POLICY_GROUPS
from embodied_demo.behavior1k.server import BehaviorWebSocketPolicyServer

from pipelines.lerobot.behavior1k.adapter import (
    BehaviorLeRobotAdapterError,
    load_behavior_view,
)
from pipelines.lerobot.behavior1k.infer import (
    _resolve_pretrained_dir,
    load_pi05_runtime,
)
from pipelines.lerobot.behavior1k.serving import (
    EvaluatorObservationKeys,
    Pi05EvaluatorPolicy,
)

LEROBOT_REQUIRED_COMMIT = "e40b58a8dfa9e7b86918c374791599d070518d11"
LEROBOT_REQUIRED_VERSION = "0.6.1"
BEHAVIOR_PROTOCOL_COMMIT = "26f2c7ef7b9cf96bd0414f81e1e751e493762779"
OPENPI_REFERENCE_COMMIT = "0cc8e355f7bac0976db1cc3139b1ff0379feea60"


def _mapping(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise BehaviorLeRobotAdapterError(f"{name} must be a YAML mapping")
    return value


def load_server_config(path: Path) -> dict[str, Any]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise BehaviorLeRobotAdapterError(
            f"cannot read PI0.5 server config {path}: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise BehaviorLeRobotAdapterError("server config root must be a mapping")
    if config.get("backend") != "lerobot_pi05":
        raise BehaviorLeRobotAdapterError("backend must be lerobot_pi05")
    for section in (
        "paths",
        "server",
        "policy",
        "runtime",
        "observation",
        "dependencies",
    ):
        _mapping(config, section)
    return config


def configure_runtime_environment(
    config: dict[str, Any],
    *,
    project_root: Path = _PROJECT_ROOT,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply the portable project-local Hugging Face cache contract."""

    runtime = _mapping(config, "runtime")
    hf_home = _resolve_path(project_root, runtime.get("hf_home", "hf_cache"))
    selected = {
        "HF_HOME": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "HF_DATASETS_CACHE": str(hf_home / "datasets"),
    }
    target_environment = os.environ if environ is None else environ
    if bool(runtime.get("direct_cuda_load", False)):
        selected["BEHAVIOR1K_PI05_DIRECT_CUDA_LOAD"] = "1"
    else:
        target_environment.pop("BEHAVIOR1K_PI05_DIRECT_CUDA_LOAD", None)
    if bool(runtime.get("offline", True)):
        selected.update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    target_environment.update(selected)
    return selected


def _resolve_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _git_output(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise BehaviorLeRobotAdapterError(
            f"cannot inspect LeRobot checkout {checkout}: {detail}"
        )
    return result.stdout.strip()


def verify_lerobot_checkout(
    checkout: Path,
    *,
    required_commit: str,
    require_clean: bool,
) -> None:
    if not checkout.is_dir():
        raise BehaviorLeRobotAdapterError(
            f"pinned LeRobot checkout does not exist: {checkout}"
        )
    actual_commit = _git_output(checkout, "rev-parse", "HEAD")
    if actual_commit != required_commit:
        raise BehaviorLeRobotAdapterError(
            "LeRobot revision mismatch: "
            f"expected {required_commit}, got {actual_commit}"
        )
    if require_clean:
        dirty = _git_output(
            checkout,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
        if dirty:
            raise BehaviorLeRobotAdapterError(
                "LeRobot checkout has modified tracked files; use a clean pinned checkout"
            )


def _validate_pins(dependencies: dict[str, Any]) -> tuple[Path, bool]:
    lerobot = dependencies.get("lerobot")
    if not isinstance(lerobot, dict):
        raise BehaviorLeRobotAdapterError(
            "dependencies.lerobot must be a mapping"
        )
    declared_commit = str(lerobot.get("commit", ""))
    declared_version = str(lerobot.get("version", ""))
    if declared_commit != LEROBOT_REQUIRED_COMMIT:
        raise BehaviorLeRobotAdapterError(
            "server config must pin the verified LeRobot commit "
            f"{LEROBOT_REQUIRED_COMMIT}"
        )
    if declared_version != LEROBOT_REQUIRED_VERSION:
        raise BehaviorLeRobotAdapterError(
            f"server config must pin LeRobot {LEROBOT_REQUIRED_VERSION}"
        )

    protocol = dependencies.get("protocol_references")
    if not isinstance(protocol, dict):
        raise BehaviorLeRobotAdapterError(
            "dependencies.protocol_references must be a mapping"
        )
    if str(protocol.get("behavior_v3_9_1_commit", "")) != BEHAVIOR_PROTOCOL_COMMIT:
        raise BehaviorLeRobotAdapterError(
            "BEHAVIOR protocol reference commit does not match v3.9.1"
        )
    if str(protocol.get("openpi_commit", "")) != OPENPI_REFERENCE_COMMIT:
        raise BehaviorLeRobotAdapterError(
            "OpenPI protocol reference commit does not match the inspected adapter"
        )
    return (
        _resolve_path(_PROJECT_ROOT, lerobot.get("checkout", "upstreams/lerobot")),
        bool(lerobot.get("require_clean_checkout", True)),
    )


def resolve_server_inputs(
    config: dict[str, Any],
    *,
    checkpoint_override: Path | None = None,
    host_override: str | None = None,
    port_override: int | None = None,
) -> dict[str, Any]:
    paths = _mapping(config, "paths")
    policy = _mapping(config, "policy")
    server = _mapping(config, "server")
    observation = _mapping(config, "observation")
    dependencies = _mapping(config, "dependencies")

    checkout, require_clean = _validate_pins(dependencies)
    verify_lerobot_checkout(
        checkout,
        required_commit=LEROBOT_REQUIRED_COMMIT,
        require_clean=require_clean,
    )
    view_dir = _resolve_path(_PROJECT_ROOT, paths.get("view_dir"))
    checkpoint = _resolve_path(
        _PROJECT_ROOT,
        checkpoint_override or paths.get("checkpoint"),
    )
    pretrained_dir = _resolve_pretrained_dir(checkpoint)
    stats_path = paths.get("view_stats")
    resolved_stats_path = (
        _resolve_path(_PROJECT_ROOT, stats_path) if stats_path else None
    )
    view = load_behavior_view(view_dir, stats_path=resolved_stats_path)

    device = str(policy.get("device", "cuda"))
    if not device.startswith("cuda"):
        raise BehaviorLeRobotAdapterError(
            "the real PI0.5 evaluator server requires a CUDA device"
        )
    num_inference_steps = policy.get("num_inference_steps")
    if num_inference_steps is not None and int(num_inference_steps) <= 0:
        raise BehaviorLeRobotAdapterError(
            "policy.num_inference_steps must be positive when set"
        )
    execution_horizon = int(server.get("execution_horizon", 16))
    if execution_horizon <= 0:
        raise BehaviorLeRobotAdapterError(
            "server.execution_horizon must be positive"
        )
    port = int(port_override if port_override is not None else server.get("port", 8000))
    if port <= 0 or port > 65535:
        raise BehaviorLeRobotAdapterError("server port must be between 1 and 65535")
    host = str(host_override or server.get("host", "0.0.0.0")).strip()
    if not host:
        raise BehaviorLeRobotAdapterError("server host must not be empty")

    camera_sensors = observation.get("camera_sensors")
    if not isinstance(camera_sensors, dict):
        raise BehaviorLeRobotAdapterError(
            "observation.camera_sensors must be a mapping"
        )
    observation_keys = EvaluatorObservationKeys.from_robot_config(
        robot_name=str(observation.get("robot_name", "robot_r1")),
        head_sensor=str(camera_sensors.get("head", "")),
        left_wrist_sensor=str(camera_sensors.get("left_wrist", "")),
        right_wrist_sensor=str(camera_sensors.get("right_wrist", "")),
    )
    return {
        "view": view,
        "view_dir": view_dir,
        "stats_path": resolved_stats_path,
        "checkpoint": pretrained_dir,
        "device": device,
        "num_inference_steps": (
            int(num_inference_steps) if num_inference_steps is not None else None
        ),
        "host": host,
        "port": port,
        "execution_horizon": execution_horizon,
        "observation_keys": observation_keys,
        "lerobot_checkout": checkout,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="启动真实 LeRobot PI0.5 checkpoint 的 BEHAVIOR-1K policy server。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT
        / "experiments/lerobot/pi05_behavior1k_task0/server.yaml",
        help="中文注释 YAML；默认使用 Task 0 server 配置。",
    )
    parser.add_argument("--checkpoint", type=Path, help="临时覆盖 checkpoint 路径。")
    parser.add_argument("--host", help="临时覆盖监听地址。")
    parser.add_argument("--port", type=int, help="临时覆盖监听端口。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验 view、checkpoint、协议 pins 和 LeRobot checkout，不加载 GPU 模型。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_server_config(args.config.resolve())
        runtime_environment = configure_runtime_environment(config)
        resolved = resolve_server_inputs(
            config,
            checkpoint_override=args.checkpoint,
            host_override=args.host,
            port_override=args.port,
        )
        dry_payload = {
            "backend": "lerobot_pi05",
            "view_dir": str(resolved["view_dir"]),
            "checkpoint": str(resolved["checkpoint"]),
            "dataset_root": str(resolved["view"].root),
            "task_instruction": resolved["view"].task_instruction,
            "host": resolved["host"],
            "port": resolved["port"],
            "execution_horizon": resolved["execution_horizon"],
            "action_dim": ACTION_DIM,
            "lerobot_checkout": str(resolved["lerobot_checkout"]),
            "lerobot_commit": LEROBOT_REQUIRED_COMMIT,
            "behavior_protocol_commit": BEHAVIOR_PROTOCOL_COMMIT,
            "openpi_reference_commit": OPENPI_REFERENCE_COMMIT,
            "gpu_model_loaded": False,
            "runtime_environment": runtime_environment,
        }
        if args.dry_run:
            print("BEHAVIOR1K_PI05_SERVER_DRY_RUN_OK")
            print(json.dumps(dry_payload, ensure_ascii=False, indent=2))
            return 0

        runtime, runtime_view = load_pi05_runtime(
            view_dir=resolved["view_dir"],
            stats_path=resolved["stats_path"],
            checkpoint=resolved["checkpoint"],
            device=resolved["device"],
        )
        policy = Pi05EvaluatorPolicy(
            runtime,
            observation_keys=resolved["observation_keys"],
            task_instruction=runtime_view.task_instruction,
            num_inference_steps=resolved["num_inference_steps"],
        )
        metadata = {
            "schema_version": "1.0",
            "backend": "lerobot",
            "policy_type": "pi05",
            "checkpoint": str(runtime.pretrained_dir),
            "task_instruction": runtime_view.task_instruction,
            "action_dim": ACTION_DIM,
            "action_groups": [name for name, _, _ in POLICY_GROUPS],
            "execution_horizon": resolved["execution_horizon"],
            "lerobot_commit": LEROBOT_REQUIRED_COMMIT,
            "behavior_protocol_commit": BEHAVIOR_PROTOCOL_COMMIT,
        }
        server = BehaviorWebSocketPolicyServer(
            policy,
            host=resolved["host"],
            port=resolved["port"],
            metadata=metadata,
            action_dim=ACTION_DIM,
            execution_horizon=resolved["execution_horizon"],
        )
        logging.info(
            "BEHAVIOR1K_PI05_MODEL_READY checkpoint=%s task=%s",
            runtime.pretrained_dir,
            runtime_view.task_instruction,
        )
        server.serve_forever()
        return 0
    except (BehaviorLeRobotAdapterError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise SystemExit(main())
