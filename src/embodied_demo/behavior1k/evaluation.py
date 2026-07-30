from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import Field, ValidationError, model_validator

from embodied_demo.config import compose_yaml
from embodied_demo.errors import PipelineError, SchemaValidationError
from embodied_demo.schemas.base import StrictModel

REQUIRED_BEHAVIOR_TAG = "v3.9.1"
REQUIRED_BEHAVIOR_COMMIT = "26f2c7ef7b9cf96bd0414f81e1e751e493762779"
DEFAULT_EVALUATOR_MODULE = "omnigibson.eval.eval"
PUBLIC_TEST_INSTANCE_COUNT = 20


class BehaviorEvaluationError(PipelineError):
    """Expected evaluator orchestration failure with actionable context."""


class BehaviorRuntimeConfig(StrictModel):
    # Environment variables override the portable repository defaults. This keeps
    # public YAML free of cluster-specific absolute paths.
    checkout: str = "upstreams/BEHAVIOR-1K"
    checkout_env: str = "BEHAVIOR1K_REPO_ROOT"
    python: str = "python"
    python_env: str = "BEHAVIOR1K_PYTHON"
    required_tag: str = REQUIRED_BEHAVIOR_TAG
    required_commit: str = REQUIRED_BEHAVIOR_COMMIT
    evaluator_module: str = DEFAULT_EVALUATOR_MODULE
    require_clean_checkout: bool = True

    @model_validator(mode="after")
    def enforce_supported_release(self) -> "BehaviorRuntimeConfig":
        if self.required_tag != REQUIRED_BEHAVIOR_TAG:
            raise ValueError(f"behavior.required_tag must be {REQUIRED_BEHAVIOR_TAG}")
        if self.required_commit != REQUIRED_BEHAVIOR_COMMIT:
            raise ValueError(
                "behavior.required_commit must pin the official BEHAVIOR-1K "
                f"{REQUIRED_BEHAVIOR_TAG} commit {REQUIRED_BEHAVIOR_COMMIT}"
            )
        if self.evaluator_module != DEFAULT_EVALUATOR_MODULE:
            raise ValueError(f"behavior.evaluator_module must be {DEFAULT_EVALUATOR_MODULE}")
        return self


class PolicyEndpointConfig(StrictModel):
    url: str = "ws://127.0.0.1:8000"
    require_healthz: bool = True
    healthz_timeout_seconds: float = Field(default=60.0, gt=0, le=3600)

    @model_validator(mode="after")
    def validate_policy_url(self) -> "PolicyEndpointConfig":
        parse_policy_url(self.url)
        return self


class OfficialEvaluationConfig(StrictModel):
    task_name: str = Field(min_length=1)
    mode: Literal["public_test"] = "public_test"
    instance_indices: list[int] = Field(default_factory=lambda: [0], min_length=1)
    num_rollouts: int = Field(default=1, ge=1, le=100)
    max_steps: int | None = Field(default=None, ge=1)
    env_wrapper: str = "omnigibson.eval.wrappers.RGBDFullResWrapper"
    robot_config: str | None = None
    robot_config_env: str = "BEHAVIOR1K_ROBOT_CONFIG"
    headless: bool = True
    write_video: bool = True
    video_fps: int = Field(default=30, ge=1, le=240)

    @model_validator(mode="after")
    def validate_public_indices(self) -> "OfficialEvaluationConfig":
        if len(set(self.instance_indices)) != len(self.instance_indices):
            raise ValueError("evaluation.instance_indices must be unique")
        invalid = [
            index
            for index in self.instance_indices
            if not 0 <= index < PUBLIC_TEST_INSTANCE_COUNT
        ]
        if invalid:
            raise ValueError(
                "public_test instance indices must be in [0, 19]; "
                f"invalid values: {invalid}"
            )
        return self


class EvaluationOutputConfig(StrictModel):
    directory: str
    resume: bool = True


class BehaviorEvaluatorConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    behavior: BehaviorRuntimeConfig = Field(default_factory=BehaviorRuntimeConfig)
    policy: PolicyEndpointConfig = Field(default_factory=PolicyEndpointConfig)
    evaluation: OfficialEvaluationConfig
    output: EvaluationOutputConfig


@dataclass(frozen=True)
class BehaviorVersionInfo:
    checkout: Path
    head: str
    tag: str
    clean: bool


@dataclass(frozen=True)
class ResolvedEvaluationPlan:
    config: BehaviorEvaluatorConfig
    config_path: Path
    config_sources: tuple[Path, ...]
    project_root: Path
    checkout: Path
    python: str
    output_dir: Path
    robot_config: Path | None
    policy_host: str
    policy_port: int
    fingerprint: str
    version: BehaviorVersionInfo


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_policy_url(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if parsed.scheme != "ws":
        raise ValueError(
            "policy.url must use ws:// because the official v3.9.1 evaluator CLI "
            "only forwards host and port"
        )
    if not parsed.hostname or parsed.port is None:
        raise ValueError("policy.url must include a hostname and explicit port")
    if parsed.username or parsed.password:
        raise ValueError("policy.url must not contain credentials")
    if parsed.path not in ("", "/") or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("policy.url must not contain a path, query, or fragment")
    return parsed.hostname, parsed.port


def load_evaluator_config(
    path: str | Path,
) -> tuple[BehaviorEvaluatorConfig, tuple[Path, ...]]:
    source = Path(path).expanduser().resolve()
    payload, sources = compose_yaml(source)
    try:
        config = BehaviorEvaluatorConfig.model_validate(payload)
    except ValidationError as exc:
        raise SchemaValidationError(f"schema validation failed for {source}:\n{exc}") from exc
    return config, tuple(sources)


def _resolve_path(
    value: str | None,
    env_name: str | None,
    project_root: Path,
) -> Path | None:
    selected = os.environ.get(env_name, "").strip() if env_name else ""
    if not selected:
        selected = (value or "").strip()
    if not selected:
        return None
    path = Path(selected).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _resolve_python(value: str, env_name: str) -> str:
    selected = os.environ.get(env_name, "").strip() or value
    selected_path = Path(selected).expanduser()
    if selected_path.is_absolute() or "/" in selected:
        # Preserve a virtualenv / conda symlink path. Resolving it to the base
        # interpreter can change sys.prefix and silently drop the evaluator env.
        absolute = Path(os.path.abspath(selected_path))
        if not absolute.is_file() or not os.access(absolute, os.X_OK):
            raise BehaviorEvaluationError(f"BEHAVIOR Python is not executable: {absolute}")
        return str(absolute)
    executable = shutil.which(selected)
    if executable is None:
        raise BehaviorEvaluationError(f"BEHAVIOR Python command is not on PATH: {selected}")
    return executable


def _git(checkout: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise BehaviorEvaluationError(
            f"git {' '.join(args)} failed for {checkout}: {detail or 'unknown error'}"
        )
    return process.stdout.strip()


def verify_behavior_checkout(
    checkout: str | Path,
    runtime: BehaviorRuntimeConfig,
) -> BehaviorVersionInfo:
    root = Path(checkout).expanduser().resolve()
    if not (root / ".git").exists():
        raise BehaviorEvaluationError(f"BEHAVIOR-1K checkout is missing or not a Git checkout: {root}")
    evaluator = root / "OmniGibson/omnigibson/eval/eval.py"
    if not evaluator.is_file():
        raise BehaviorEvaluationError(f"official evaluator entrypoint is missing: {evaluator}")

    head = _git(root, "rev-parse", "HEAD")
    if head != runtime.required_commit:
        raise BehaviorEvaluationError(
            "BEHAVIOR-1K version mismatch: "
            f"required {runtime.required_tag} ({runtime.required_commit}), got {head}"
        )

    try:
        tag_commit = _git(root, "rev-parse", f"refs/tags/{runtime.required_tag}^{{commit}}")
    except BehaviorEvaluationError as exc:
        raise BehaviorEvaluationError(
            f"required Git tag is not available in checkout: {runtime.required_tag}"
        ) from exc
    if tag_commit != runtime.required_commit:
        raise BehaviorEvaluationError(
            f"tag {runtime.required_tag} resolves to {tag_commit}, "
            f"expected {runtime.required_commit}"
        )

    tracked_status = _git(root, "status", "--porcelain", "--untracked-files=no")
    clean = not tracked_status
    if runtime.require_clean_checkout and not clean:
        raise BehaviorEvaluationError(
            f"BEHAVIOR-1K tracked source is modified; refusing reproducibility claim: {root}"
        )
    return BehaviorVersionInfo(
        checkout=root,
        head=head,
        tag=runtime.required_tag,
        clean=clean,
    )


def _config_fingerprint(
    config: BehaviorEvaluatorConfig,
    checkout: Path,
    python: str,
    robot_config: Path | None,
    policy_url: str,
) -> str:
    config_payload = config.model_dump(mode="json")
    # Output path and resume are orchestration controls, not evaluation
    # semantics. Excluding them lets a completed non-resume run be reopened
    # explicitly with resume=true while all model/evaluator parameters remain
    # identical.
    config_payload.pop("output", None)
    payload = {
        "config": config_payload,
        "resolved": {
            "checkout": str(checkout),
            "python": python,
            "robot_config": str(robot_config) if robot_config else None,
            "policy_url": policy_url,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_evaluation_plan(
    config_path: str | Path,
    *,
    project_root: str | Path,
    output_dir: str | Path | None = None,
    policy_url: str | None = None,
    instance_indices: list[int] | None = None,
    resume: bool | None = None,
) -> ResolvedEvaluationPlan:
    config, sources = load_evaluator_config(config_path)
    payload = config.model_dump(mode="python")
    if policy_url is not None:
        payload["policy"]["url"] = policy_url
    if instance_indices is not None:
        payload["evaluation"]["instance_indices"] = instance_indices
    if output_dir is not None:
        payload["output"]["directory"] = str(output_dir)
    if resume is not None:
        payload["output"]["resume"] = resume
    try:
        config = BehaviorEvaluatorConfig.model_validate(payload)
    except ValidationError as exc:
        raise SchemaValidationError(f"invalid evaluator override:\n{exc}") from exc

    root = Path(project_root).expanduser().resolve()
    checkout = _resolve_path(config.behavior.checkout, config.behavior.checkout_env, root)
    assert checkout is not None
    python = _resolve_python(config.behavior.python, config.behavior.python_env)
    resolved_output = _resolve_path(config.output.directory, None, root)
    assert resolved_output is not None
    robot_config = _resolve_path(
        config.evaluation.robot_config,
        config.evaluation.robot_config_env,
        root,
    )
    if robot_config is not None and not robot_config.is_file():
        raise BehaviorEvaluationError(f"robot config does not exist: {robot_config}")

    host, port = parse_policy_url(config.policy.url)
    version = verify_behavior_checkout(checkout, config.behavior)
    fingerprint = _config_fingerprint(config, checkout, python, robot_config, config.policy.url)
    return ResolvedEvaluationPlan(
        config=config,
        config_path=Path(config_path).expanduser().resolve(),
        config_sources=sources,
        project_root=root,
        checkout=checkout,
        python=python,
        output_dir=resolved_output,
        robot_config=robot_config,
        policy_host=host,
        policy_port=port,
        fingerprint=fingerprint,
        version=version,
    )


def build_evaluator_command(plan: ResolvedEvaluationPlan, instance_index: int) -> list[str]:
    cfg = plan.config.evaluation
    command = [
        plan.python,
        "-m",
        plan.config.behavior.evaluator_module,
        "--task-name",
        cfg.task_name,
        "--host",
        plan.policy_host,
        "--port",
        str(plan.policy_port),
        "--policy",
        "websocket",
        "--mode",
        cfg.mode,
        "--instance-indices",
        str(instance_index),
        "--num-rollouts",
        str(cfg.num_rollouts),
        "--env-wrapper",
        cfg.env_wrapper,
        "--output-dir",
        str(plan.output_dir),
        "--video-fps",
        str(cfg.video_fps),
        "--write-video" if cfg.write_video else "--no-write-video",
        "--headless" if cfg.headless else "--no-headless",
    ]
    if cfg.max_steps is not None:
        command.extend(["--max-steps", str(cfg.max_steps)])
    if plan.robot_config is not None:
        command.extend(["--robot-config", str(plan.robot_config)])
    return command


def format_dry_run(plan: ResolvedEvaluationPlan) -> str:
    lines = [
        "BEHAVIOR1K_EVAL_DRY_RUN_OK",
        f"behavior_version={plan.version.tag}",
        f"behavior_commit={plan.version.head}",
        f"checkout={plan.checkout}",
        f"python={plan.python}",
        f"policy_url={plan.config.policy.url}",
        f"output_dir={plan.output_dir}",
        f"resume={str(plan.config.output.resume).lower()}",
    ]
    for index in plan.config.evaluation.instance_indices:
        lines.append(f"instance[{index}]={shlex.join(build_evaluator_command(plan, index))}")
    return "\n".join(lines)


def _healthz_url(plan: ResolvedEvaluationPlan) -> str:
    return f"http://{plan.policy_host}:{plan.policy_port}/healthz"


def _wait_for_healthz(plan: ResolvedEvaluationPlan) -> None:
    if not plan.config.policy.require_healthz:
        return
    url = _healthz_url(plan)
    deadline = time.monotonic() + plan.config.policy.healthz_timeout_seconds
    last_error = "no response"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        try:
            with opener.open(url, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise BehaviorEvaluationError(
        f"policy health check did not pass within "
        f"{plan.config.policy.healthz_timeout_seconds:g}s: {url} ({last_error})"
    )


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _new_state(plan: ResolvedEvaluationPlan) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "config_fingerprint": plan.fingerprint,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "completed": {},
    }


def _load_or_initialize_state(plan: ResolvedEvaluationPlan) -> dict[str, Any]:
    state_path = plan.output_dir / "orchestrator_state.json"
    if not plan.output_dir.exists():
        plan.output_dir.mkdir(parents=True)
    entries = list(plan.output_dir.iterdir())

    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BehaviorEvaluationError(f"cannot read evaluator state: {state_path}: {exc}") from exc
        if not plan.config.output.resume:
            raise BehaviorEvaluationError(
                f"output directory already contains a prior run and resume=false: {plan.output_dir}"
            )
        if state.get("config_fingerprint") != plan.fingerprint:
            raise BehaviorEvaluationError(
                "resume configuration does not match the existing evaluator state; "
                "use a new output directory"
            )
        if not isinstance(state.get("completed"), dict):
            raise BehaviorEvaluationError(f"invalid completed map in evaluator state: {state_path}")
        return state

    if entries:
        raise BehaviorEvaluationError(
            "refusing to mix evaluator outputs without orchestrator_state.json: "
            f"{plan.output_dir}"
        )
    state = _new_state(plan)
    _atomic_write_json(state_path, state)
    return state


def _result_file_is_valid(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    required = ("task", "instance_id", "rollout_id", "steps", "success")
    return (
        all(field in payload for field in required)
        and isinstance(payload["task"], str)
        and isinstance(payload["instance_id"], int)
        and isinstance(payload["rollout_id"], int)
        and isinstance(payload["steps"], int)
        and isinstance(payload["success"], bool)
    )


def _completion_is_valid(
    output_dir: Path,
    completion: Any,
    expected_rollouts: int,
) -> bool:
    if not isinstance(completion, dict):
        return False
    files = completion.get("result_files")
    if not isinstance(files, list) or len(files) < expected_rollouts:
        return False
    return all(
        isinstance(relative, str)
        and _result_file_is_valid(output_dir / relative)
        for relative in files
    )


def _file_snapshot(
    directory: Path,
    pattern: str,
) -> dict[Path, tuple[int, int]]:
    if not directory.is_dir():
        return {}
    return {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in directory.glob(pattern)
        if path.is_file()
    }


def _changed_files(
    before: dict[Path, tuple[int, int]],
    after: dict[Path, tuple[int, int]],
) -> list[Path]:
    return sorted(path for path, signature in after.items() if before.get(path) != signature)


def _execute_command(command: list[str], cwd: Path, log_path: Path) -> int:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise BehaviorEvaluationError(
                f"cannot start official evaluator command: {shlex.join(command)}: {exc}"
            ) from exc
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()


def _write_resolved_metadata(plan: ResolvedEvaluationPlan) -> None:
    resolved = {
        "schema_version": "1.0",
        "config_sources": [str(path) for path in plan.config_sources],
        "behavior": {
            "checkout": str(plan.checkout),
            "required_tag": plan.version.tag,
            "required_commit": plan.version.head,
            "tracked_source_clean": plan.version.clean,
            "python": plan.python,
        },
        "policy": {
            **plan.config.policy.model_dump(mode="json"),
            "healthz_url": _healthz_url(plan),
        },
        "evaluation": {
            **plan.config.evaluation.model_dump(mode="json"),
            "robot_config": str(plan.robot_config) if plan.robot_config else None,
        },
        "output": {
            "directory": str(plan.output_dir),
            "resume": plan.config.output.resume,
        },
        "config_fingerprint": plan.fingerprint,
    }
    _atomic_write_text(
        plan.output_dir / "resolved_config.yaml",
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
    )
    environment = {
        "created_at": _utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "orchestrator_python": sys.executable,
        "behavior_python": plan.python,
        "behavior_commit": plan.version.head,
        "behavior_tag": plan.version.tag,
    }
    _atomic_write_json(plan.output_dir / "environment.json", environment)


def summarize_official_results(
    output_dir: str | Path,
    *,
    task_name: str,
    requested_indices: list[int],
    num_rollouts: int,
    state: dict[str, Any],
) -> dict[str, Any]:
    root = Path(output_dir)
    completed_indices: list[int] = []
    result_paths: list[Path] = []
    for index in requested_indices:
        completion = state.get("completed", {}).get(str(index))
        if not _completion_is_valid(root, completion, num_rollouts):
            continue
        completed_indices.append(index)
        result_paths.extend(root / relative for relative in completion["result_files"])

    unique_paths = sorted(set(result_paths))
    results: list[dict[str, Any]] = []
    for path in unique_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        results.append(payload)

    successes = [result.get("success") for result in results if isinstance(result.get("success"), bool)]
    q_scores: list[float] = []
    for result in results:
        q_score = result.get("q_score")
        final = q_score.get("final") if isinstance(q_score, dict) else None
        if isinstance(final, (int, float)):
            q_scores.append(float(final))

    expected_results = len(requested_indices) * num_rollouts
    summary = {
        "schema_version": "1.0",
        "generated_at": _utc_now(),
        "authority": "official BEHAVIOR-1K JSON files; this file is an audit summary only",
        "task_name": task_name,
        "requested_public_instance_indices": requested_indices,
        "completed_public_instance_indices": completed_indices,
        "num_rollouts_per_instance": num_rollouts,
        "expected_result_count": expected_results,
        "official_result_count": len(results),
        "complete": (
            completed_indices == requested_indices
            and len(results) >= expected_results
        ),
        "success_count": sum(successes),
        "success_rate": (sum(successes) / len(successes)) if successes else None,
        "mean_final_q_score": (sum(q_scores) / len(q_scores)) if q_scores else None,
        "official_result_files": [str(path.relative_to(root)) for path in unique_paths],
    }
    return summary


def run_evaluation(plan: ResolvedEvaluationPlan) -> dict[str, Any]:
    state = _load_or_initialize_state(plan)
    _write_resolved_metadata(plan)
    state_path = plan.output_dir / "orchestrator_state.json"
    attempts_path = plan.output_dir / "attempts.jsonl"
    log_path = plan.output_dir / "evaluator_stdout.log"

    pending_indices = [
        index
        for index in plan.config.evaluation.instance_indices
        if not _completion_is_valid(
            plan.output_dir,
            state["completed"].get(str(index)),
            plan.config.evaluation.num_rollouts,
        )
    ]
    if pending_indices:
        _wait_for_healthz(plan)
    try:
        for index in plan.config.evaluation.instance_indices:
            existing = state["completed"].get(str(index))
            if _completion_is_valid(
                plan.output_dir,
                existing,
                plan.config.evaluation.num_rollouts,
            ):
                print(f"BEHAVIOR1K_EVAL_RESUME_SKIP public_index={index}")
                continue

            command = build_evaluator_command(plan, index)
            before_json = _file_snapshot(plan.output_dir / "json", "*.json")
            before_video = _file_snapshot(plan.output_dir / "videos", "*.mp4")
            attempt_id = f"public-{index}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')}"
            _append_jsonl(
                attempts_path,
                {
                    "event": "start",
                    "attempt_id": attempt_id,
                    "created_at": _utc_now(),
                    "public_instance_index": index,
                    "command": command,
                },
            )
            return_code = _execute_command(command, plan.checkout, log_path)
            _append_jsonl(
                attempts_path,
                {
                    "event": "finish",
                    "attempt_id": attempt_id,
                    "created_at": _utc_now(),
                    "public_instance_index": index,
                    "return_code": return_code,
                },
            )
            if return_code != 0:
                raise BehaviorEvaluationError(
                    f"official evaluator failed for public index {index} with status {return_code}; "
                    f"see {log_path}"
                )

            after_json = _file_snapshot(plan.output_dir / "json", "*.json")
            changed_json = _changed_files(before_json, after_json)
            valid_json = [path for path in changed_json if _result_file_is_valid(path)]
            if len(valid_json) < plan.config.evaluation.num_rollouts:
                raise BehaviorEvaluationError(
                    "official evaluator exited successfully but did not produce the expected "
                    f"{plan.config.evaluation.num_rollouts} new/updated result JSON files "
                    f"for public index {index}"
                )

            changed_videos: list[Path] = []
            if plan.config.evaluation.write_video:
                after_video = _file_snapshot(plan.output_dir / "videos", "*.mp4")
                changed_videos = _changed_files(before_video, after_video)
                if len(changed_videos) < plan.config.evaluation.num_rollouts:
                    raise BehaviorEvaluationError(
                        "official evaluator did not produce the expected video files "
                        f"for public index {index}"
                    )

            state["completed"][str(index)] = {
                "completed_at": _utc_now(),
                "result_files": [
                    str(path.relative_to(plan.output_dir)) for path in valid_json
                ],
                "video_files": [
                    str(path.relative_to(plan.output_dir)) for path in changed_videos
                ],
            }
            state["updated_at"] = _utc_now()
            _atomic_write_json(state_path, state)
    finally:
        summary = summarize_official_results(
            plan.output_dir,
            task_name=plan.config.evaluation.task_name,
            requested_indices=plan.config.evaluation.instance_indices,
            num_rollouts=plan.config.evaluation.num_rollouts,
            state=state,
        )
        _atomic_write_json(plan.output_dir / "summary.json", summary)

    return summary


__all__ = [
    "BehaviorEvaluationError",
    "BehaviorEvaluatorConfig",
    "BehaviorVersionInfo",
    "ResolvedEvaluationPlan",
    "REQUIRED_BEHAVIOR_COMMIT",
    "REQUIRED_BEHAVIOR_TAG",
    "PUBLIC_TEST_INSTANCE_COUNT",
    "build_evaluator_command",
    "format_dry_run",
    "load_evaluator_config",
    "parse_policy_url",
    "resolve_evaluation_plan",
    "run_evaluation",
    "summarize_official_results",
    "verify_behavior_checkout",
]
