#!/usr/bin/env python3
from __future__ import annotations

"""百舸 PyTorchJob 原生分布式 launcher。

这个脚本只做一件事：读取百舸自动注入的节点级环境变量，把它们转换成
项目内 LeRobot / FastWAM 两条训练链路已经使用的 YAML 配置环境变量。

正式 PyTorchJob 里，Master 和 Worker 会同时执行同一条 command，因此不需要
ssh 到 worker，也不需要用户在每台机器上手动执行命令。
"""

import argparse
import os
import re
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - cluster-side error path.
    raise SystemExit("ERROR: PyYAML is required. Install with: python -m pip install PyYAML") from exc


SUPPORTED_BACKENDS = {"lerobot", "fastwam"}
PYTORCHJOB_ENV_NAMES = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "NPROC_PER_NODE",
)
FASTWAM_PROFILES = {"smoke", "pilot", "full"}


def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "scripts").is_dir():
            return path
    raise SystemExit(f"ERROR: cannot locate project root from {start}")


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: YAML root must be a mapping: {path}")
    return data


def int_env(name: str, default: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        if default is None:
            raise SystemExit(f"ERROR: missing required Baige/PyTorchJob env: {name}")
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {name} must be an integer, got {value!r}") from exc


def infer_nproc_per_node() -> int:
    env_value = os.environ.get("NPROC_PER_NODE")
    if env_value:
        return int_env("NPROC_PER_NODE")
    try:
        import torch

        count = torch.cuda.device_count()
        if count > 0:
            return count
    except Exception:
        pass
    raise SystemExit("ERROR: NPROC_PER_NODE is not set and torch.cuda.device_count() failed or returned 0")


def require_pytorchjob_env() -> None:
    missing = [name for name in PYTORCHJOB_ENV_NAMES if not os.environ.get(name, "").strip()]
    if missing:
        raise SystemExit(
            "ERROR: missing required Baige PyTorchJob env: " + ", ".join(missing)
        )


def safe_run_id_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return normalized or "baige-job"


def experiment_name(config: dict[str, Any], config_path: Path) -> str:
    section = config.get("experiment") or {}
    if not isinstance(section, dict):
        raise SystemExit("ERROR: experiment section must be a mapping")
    return str(section.get("name") or config_path.parent.name)


def stable_run_id(config: dict[str, Any], config_path: Path, backend: str) -> str:
    env_name = "LEROBOT_RUN_ID" if backend == "lerobot" else "FASTWAM_RUN_ID"
    if os.environ.get(env_name):
        return str(os.environ[env_name])
    if os.environ.get("BAIGE_RUN_ID"):
        return str(os.environ["BAIGE_RUN_ID"])

    section = config.get("experiment") or {}
    configured = section.get("run_id") if isinstance(section, dict) else ""
    if configured:
        return str(configured)

    job_id = os.environ.get("AIHC_JOB_ID") or os.environ.get("JOB_ID")
    if job_id:
        return (
            f"{experiment_name(config, config_path)}_"
            f"{safe_run_id_component(job_id)}"
        )

    # 百舸默认 MASTER_ADDR={任务名称}-master-0，所有节点一致，因此它可以在平台
    # 没有额外注入 job id 时充当共享 run id。若任务名称会重复，应显式设置
    # BAIGE_RUN_ID，避免不同任务写到同一输出目录。
    master_addr = os.environ.get("MASTER_ADDR", "").strip()
    if master_addr:
        return (
            f"{experiment_name(config, config_path)}_"
            f"{safe_run_id_component(master_addr)}"
        )

    # 本地单机调试可以接受 hostname fallback；多机仍要求一个跨节点共享标识。
    world_size = int_env("WORLD_SIZE", 1)
    if world_size > 1:
        raise SystemExit(
            "ERROR: multi-node Baige launch requires a shared run id. "
            "Set BAIGE_RUN_ID=<same_id_on_all_nodes> or LEROBOT_RUN_ID/FASTWAM_RUN_ID."
        )
    return f"{experiment_name(config, config_path)}_{socket.gethostname()}"


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def baige_env(config: dict[str, Any], config_path: Path, backend: str) -> dict[str, str]:
    # 百舸文档定义：RANK/WORLD_SIZE 是节点级，不是全局 GPU rank。
    node_rank = int_env("RANK", 0)
    nnodes = int_env("WORLD_SIZE", 1)
    nproc_per_node = infer_nproc_per_node()
    master_addr = os.environ.get("MASTER_ADDR") or socket.gethostname()
    master_port = os.environ.get("MASTER_PORT") or "23456"
    if nnodes <= 0:
        raise SystemExit(f"ERROR: WORLD_SIZE must be positive, got {nnodes}")
    if not 0 <= node_rank < nnodes:
        raise SystemExit(
            f"ERROR: RANK must satisfy 0 <= RANK < WORLD_SIZE, got "
            f"RANK={node_rank}, WORLD_SIZE={nnodes}"
        )
    if nproc_per_node <= 0:
        raise SystemExit(
            f"ERROR: NPROC_PER_NODE must be positive, got {nproc_per_node}"
        )
    try:
        parsed_master_port = int(master_port)
    except ValueError as exc:
        raise SystemExit(
            f"ERROR: MASTER_PORT must be an integer, got {master_port!r}"
        ) from exc
    if not 1 <= parsed_master_port <= 65535:
        raise SystemExit(
            f"ERROR: MASTER_PORT must be between 1 and 65535, got {parsed_master_port}"
        )
    run_id = stable_run_id(config, config_path, backend)

    env: dict[str, str] = {
        "BAIGE_NODE_RANK": str(node_rank),
        "BAIGE_NNODES": str(nnodes),
        "BAIGE_NPROC_PER_NODE": str(nproc_per_node),
        "BAIGE_MASTER_ADDR": str(master_addr),
        "BAIGE_MASTER_PORT": str(master_port),
    }

    if backend == "lerobot":
        env.update(
            {
                "LEROBOT_RUN_ID": run_id,
                # accelerate 的 num_processes 是全局总进程数。
                "LEROBOT_NUM_PROCESSES": str(nnodes * nproc_per_node),
                "LEROBOT_NUM_MACHINES": str(nnodes),
                "LEROBOT_MACHINE_RANK": str(node_rank),
                "LEROBOT_MAIN_PROCESS_IP": str(master_addr),
                "LEROBOT_MAIN_PROCESS_PORT": str(master_port),
            }
        )
    elif backend == "fastwam":
        env.update(
            {
                "FASTWAM_RUN_ID": run_id,
                "FASTWAM_GPUS_PER_NODE": str(nproc_per_node),
                "FASTWAM_NNODES": str(nnodes),
                "FASTWAM_NODE_RANK": str(node_rank),
                "FASTWAM_MASTER_ADDR": str(master_addr),
                "FASTWAM_MASTER_PORT": str(master_port),
            }
        )
    else:
        raise AssertionError(backend)
    return env


def generated_shell_path(project_root: Path, config: dict[str, Any], config_path: Path, backend: str, env: dict[str, str]) -> Path:
    run_id = env["LEROBOT_RUN_ID"] if backend == "lerobot" else env["FASTWAM_RUN_ID"]
    node_rank = env["BAIGE_NODE_RANK"]
    return (
        project_root
        / "runs/generated_configs"
        / backend
        / experiment_name(config, config_path)
        / f"{run_id}.rank{node_rank}.sh"
    )


def backend_command(
    project_root: Path,
    config: dict[str, Any],
    config_path: Path,
    backend: str,
    env: dict[str, str],
) -> list[str]:
    if backend == "lerobot":
        runner = project_root / "scripts/lerobot/run_config.py"
    elif backend == "fastwam":
        runner = project_root / "scripts/fastwam/run_config.py"
    else:
        raise AssertionError(backend)
    return [
        sys.executable,
        str(runner),
        "--config",
        str(config_path),
        "--output-shell",
        str(generated_shell_path(project_root, config, config_path, backend, env)),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a project experiment under Baige PyTorchJob env.")
    parser.add_argument("--config", required=True, help="Path to experiment config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="只打印解析结果和底层命令，不启动训练")
    parser.add_argument("--print-command", action="store_true", help="只打印底层命令，不执行")
    parser.add_argument(
        "--profile",
        choices=sorted(FASTWAM_PROFILES),
        help="FastWAM 运行档位；也可用 BAIGE_PROFILE=smoke|pilot|full。",
    )
    parser.add_argument(
        "--require-platform-env",
        action="store_true",
        help="要求百舸注入的五个 PyTorchJob 环境变量全部存在。",
    )
    args, passthrough = parser.parse_known_args(argv)

    if args.require_platform_env:
        require_pytorchjob_env()

    config_path = Path(args.config).resolve()
    project_root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    backend = str(config.get("backend") or "").strip()
    if backend not in SUPPORTED_BACKENDS:
        raise SystemExit(f"ERROR: unsupported backend={backend!r}; expected one of {sorted(SUPPORTED_BACKENDS)}")

    env_updates = baige_env(config, config_path, backend)
    requested_profile = (
        args.profile
        or os.environ.get("BAIGE_PROFILE", "").strip()
        or os.environ.get("FASTWAM_MODE", "").strip()
    )
    if requested_profile:
        if backend != "fastwam":
            raise SystemExit("ERROR: --profile/BAIGE_PROFILE currently applies only to FastWAM")
        if requested_profile not in FASTWAM_PROFILES:
            raise SystemExit(
                "ERROR: FastWAM profile must be smoke|pilot|full, got "
                f"{requested_profile!r}"
            )
        env_updates["FASTWAM_MODE"] = requested_profile
    command = backend_command(project_root, config, config_path, backend, env_updates)
    if args.dry_run:
        command.append("--dry-run")
    command.extend(passthrough)

    print("BAIGE_NATIVE_LAUNCH", flush=True)
    print(f"  backend={backend}", flush=True)
    print(f"  host={socket.gethostname()}", flush=True)
    print(f"  config={config_path}", flush=True)
    for key in sorted(env_updates):
        print(f"  {key}={env_updates[key]}", flush=True)
    print("BAIGE_RUN_COMMAND", shell_join(command), flush=True)

    if args.print_command:
        return 0

    child_env = os.environ.copy()
    child_env.update(env_updates)
    return subprocess.call(command, cwd=project_root, env=child_env)


if __name__ == "__main__":
    raise SystemExit(main())
