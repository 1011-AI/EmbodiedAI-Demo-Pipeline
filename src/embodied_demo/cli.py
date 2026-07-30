from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from embodied_demo import __version__
from embodied_demo.errors import PipelineError, SchemaValidationError
from embodied_demo.fastwam_report import generate_fastwam_report
from embodied_demo.schemas import (
    ActionChunk,
    DatasetEvidence,
    EpisodeResult,
    EvaluationManifest,
    InferenceEvidence,
    Observation,
    TrainingEvidence,
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _command_export_schema(args: argparse.Namespace) -> int:
    destination = Path(args.output_dir).expanduser().resolve()
    schemas = {
        "observation.schema.json": Observation,
        "action_chunk.schema.json": ActionChunk,
        "episode_result.schema.json": EpisodeResult,
        "evaluation_manifest.schema.json": EvaluationManifest,
        "training_evidence.schema.json": TrainingEvidence,
        "dataset_evidence.schema.json": DatasetEvidence,
        "inference_evidence.schema.json": InferenceEvidence,
    }
    for filename, model in schemas.items():
        content = json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n"
        _write_text(destination / filename, content)
    print(f"EXPORTED {len(schemas)} schemas to {destination}")
    return 0


def _command_report_fastwam(args: argparse.Namespace) -> int:
    artifact_dir = generate_fastwam_report(
        args.run_dir,
        output_dir=args.output_dir,
        chain_config=args.chain_config,
    )
    evidence = json.loads((artifact_dir / "training_evidence.json").read_text(encoding="utf-8"))
    loss_decreased = evidence["loss_decreased"]
    loss_decreased_text = "unknown" if loss_decreased is None else str(loss_decreased).lower()
    print(f"REPORT_FASTWAM_COMPLETE {artifact_dir}")
    print(
        "SUMMARY "
        f"status={evidence['validation_status']} "
        f"loss_decreased={loss_decreased_text} "
        f"initial_loss={evidence['initial_loss']} "
        f"final_loss={evidence['final_loss']} "
        f"steps={evidence['final_step']}/{evidence['max_steps']}"
    )
    print(f"REPORT {artifact_dir / 'report.md'}")
    print(f"EVIDENCE {artifact_dir / 'training_evidence.json'}")
    return 0


def _command_behavior1k_doctor(args: argparse.Namespace) -> int:
    # Keep PyArrow and other Behavior-only dependencies outside the core import
    # path. Metadata-only checks work in the lightweight project environment.
    from embodied_demo.behavior1k.dataset import (
        load_dataset_config,
        resolve_dataset_root,
        run_dataset_doctor,
    )

    config = load_dataset_config(args.config)
    if args.scan_mode:
        config.doctor.scan_mode = args.scan_mode
    root = resolve_dataset_root(config, root_override=args.root)
    report = run_dataset_doctor(config, root=root)
    content = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    _write_text(Path(args.output).expanduser().resolve(), content)
    failed = sum(check.status == "fail" for check in report.checks)
    warnings = sum(check.status == "warning" for check in report.checks)
    print(
        "BEHAVIOR1K_DOCTOR "
        f"passed={str(report.passed).lower()} "
        f"failed={failed} warnings={warnings} "
        f"root={report.dataset_root}"
    )
    print(f"REPORT {Path(args.output).expanduser().resolve()}")
    return 0 if report.passed else 1


def _command_behavior1k_prepare_view(args: argparse.Namespace) -> int:
    from embodied_demo.behavior1k.dataset import (
        load_dataset_config,
        load_task_config,
        resolve_dataset_root,
    )
    from embodied_demo.behavior1k.view import prepare_virtual_view

    config = load_dataset_config(args.config)
    task = load_task_config(args.task_config)
    root = resolve_dataset_root(config, root_override=args.root)
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = prepare_virtual_view(
        source_root=root,
        output_dir=output_dir,
        config=config,
        task=task,
    )
    print(
        "BEHAVIOR1K_VIEW_READY "
        f"task={manifest.task.task_name} episodes={manifest.episode_count} "
        f"frames={manifest.frame_count}"
    )
    print(f"MANIFEST {output_dir / 'view_manifest.json'}")
    return 0


def _command_behavior1k_contract_smoke(args: argparse.Namespace) -> int:
    from embodied_demo.behavior1k.protocol import BehaviorPolicySession, packb, unpackb

    try:
        import numpy as np
    except ImportError as exc:
        raise PipelineError(
            "behavior1k-contract-smoke requires NumPy and MessagePack; "
            "install the Behavior extras with: pip install -e '.[behavior1k]'"
        ) from exc

    class StubPolicy:
        def __init__(self) -> None:
            self.reset_count = 0

        def predict_action_chunk(self, observation: dict[str, object]) -> object:
            horizon = int(observation.get("horizon", 4))
            return np.zeros((horizon, 23), dtype=np.float32)

        def reset(self) -> None:
            self.reset_count += 1

    policy = StubPolicy()
    session = BehaviorPolicySession(
        policy,
        metadata={"policy": "contract_stub", "action_dim": 23},
        execution_horizon=args.execution_horizon,
    )
    metadata = unpackb(session.open_frame())
    response = unpackb(session.handle_frame(packb({"horizon": args.chunk_horizon})))
    reset_response = session.handle_frame(packb({"reset": True}))
    if (
        metadata.get("action_dim") != 23
        or tuple(response["action"].shape) != (23,)
        or reset_response is not None
        or policy.reset_count != 1
    ):
        raise PipelineError("BEHAVIOR-1K policy contract smoke failed")
    print(
        "BEHAVIOR1K_CONTRACT_SMOKE_OK "
        f"action_shape={tuple(response['action'].shape)} "
        f"execution_horizon={args.execution_horizon} reset_ack=false"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embodied-demo",
        description="Contract and configuration tools for the EmbodiedAI demo pipeline.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_fastwam = subparsers.add_parser(
        "report-fastwam", help="normalize a FastWAM run into demo-chain evidence"
    )
    report_fastwam.add_argument("--run-dir", required=True, type=Path)
    report_fastwam.add_argument("--output-dir", type=Path)
    report_fastwam.add_argument(
        "--chain-config",
        type=Path,
        default=Path("demo_chains/fastwam_realrobot_v0.yaml"),
    )
    report_fastwam.set_defaults(handler=_command_report_fastwam)

    export_schema = subparsers.add_parser(
        "export-schema", help="export public contracts as JSON Schema"
    )
    export_schema.add_argument("--output-dir", type=Path, default=Path("schemas"))
    export_schema.set_defaults(handler=_command_export_schema)

    behavior_doctor = subparsers.add_parser(
        "behavior1k-doctor",
        help="validate a local BEHAVIOR-1K 2026 dataset without loading a model",
    )
    behavior_doctor.add_argument(
        "--config",
        type=Path,
        default=Path("configs/behavior1k/dataset_2026.yaml"),
    )
    behavior_doctor.add_argument("--root", type=Path)
    behavior_doctor.add_argument(
        "--scan-mode",
        choices=["metadata", "selected_task", "full_index"],
    )
    behavior_doctor.add_argument(
        "--output",
        type=Path,
        default=Path("runs/behavior1k/doctor/report.json"),
    )
    behavior_doctor.set_defaults(handler=_command_behavior1k_doctor)

    behavior_view = subparsers.add_parser(
        "behavior1k-prepare-view",
        help="write a zero-copy task/episode manifest for both model routes",
    )
    behavior_view.add_argument(
        "--config",
        type=Path,
        default=Path("configs/behavior1k/dataset_2026.yaml"),
    )
    behavior_view.add_argument(
        "--task-config",
        type=Path,
        default=Path("configs/behavior1k/tasks/turning_on_radio.yaml"),
    )
    behavior_view.add_argument("--root", type=Path)
    behavior_view.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/behavior1k/views/r1pro_policy23/turning_on_radio"),
    )
    behavior_view.set_defaults(handler=_command_behavior1k_prepare_view)

    behavior_contract = subparsers.add_parser(
        "behavior1k-contract-smoke",
        help="exercise metadata/action/reset semantics without a model or simulator",
    )
    behavior_contract.add_argument("--chunk-horizon", type=int, default=32)
    behavior_contract.add_argument("--execution-horizon", type=int, default=8)
    behavior_contract.set_defaults(handler=_command_behavior1k_contract_smoke)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (PipelineError, ValidationError) as exc:
        if isinstance(exc, ValidationError):
            exc = SchemaValidationError(str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
