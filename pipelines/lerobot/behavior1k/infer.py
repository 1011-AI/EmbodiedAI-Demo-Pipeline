"""Offline inference from a real LeRobot PI0.5 checkpoint on a Behavior view."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from embodied_demo.behavior1k.r1pro import ACTION_DIM, RGB_VIDEO_KEYS

from .adapter import (
    ACTION,
    BehaviorLeRobotAdapterError,
    adapt_lerobot_dataset,
    load_behavior_view,
)


def _resolve_pretrained_dir(path: Path) -> Path:
    candidates = (path, path / "pretrained_model")
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
    raise BehaviorLeRobotAdapterError(
        f"{path} is not a LeRobot checkpoint: expected config.json either "
        "there or under pretrained_model/"
    )


def _tensor_summary(value: Any) -> dict[str, Any]:
    array = value.detach().float().cpu().numpy()
    raw = array.tobytes()
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "first_action": array.reshape(-1, array.shape[-1])[0].tolist(),
    }


def run_inference(
    *,
    view_dir: Path,
    stats_path: Path | None,
    checkpoint: Path,
    output_dir: Path,
    sample_index: int,
    device: str,
    video_backend: str,
    num_inference_steps: int | None,
) -> Path:
    """Load dataset, checkpoint and processors, then predict one real action chunk."""

    try:
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
    except ImportError as exc:
        raise BehaviorLeRobotAdapterError(
            "offline inference requires the pinned LeRobot training environment"
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise BehaviorLeRobotAdapterError("CUDA inference requested but CUDA is unavailable")

    view = load_behavior_view(view_dir, stats_path=stats_path)
    pretrained_dir = _resolve_pretrained_dir(checkpoint.expanduser().resolve())
    base_dataset = LeRobotDataset(
        view.source_repo_id,
        root=view.root,
        episodes=list(view.episode_indices),
        revision=view.source_revision,
        video_backend=video_backend,
        return_uint8=True,
    )
    dataset = adapt_lerobot_dataset(
        base_dataset,
        view_stats=view.stats,
        video_keys=view.video_keys,
        task_instruction=view.task_instruction,
    )
    if sample_index < 0 or sample_index >= len(dataset):
        raise BehaviorLeRobotAdapterError(
            f"sample_index={sample_index} is outside [0, {len(dataset)})"
        )

    policy_config = PI05Config.from_pretrained(str(pretrained_dir))
    policy_config.pretrained_path = pretrained_dir
    policy_config.device = device
    # Infer the Behavior 23D + three-RGB feature contract from the adapted
    # metadata even when the supplied checkpoint is the generic PI0.5 base.
    policy_config.input_features = {}
    policy_config.output_features = {}
    policy = make_policy(cfg=policy_config, ds_meta=dataset.meta)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {
                    **policy.config.input_features,
                    **policy.config.output_features,
                },
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
            "device_processor": {"device": "cpu"},
        },
    )

    sample = dict(dataset[sample_index])
    # Action is a training target, not an inference input.
    sample.pop(ACTION, None)
    for key in tuple(sample):
        if key.endswith("_is_pad"):
            sample.pop(key)
    for camera_key in RGB_VIDEO_KEYS:
        image = sample[camera_key]
        if image.dtype == torch.uint8:
            sample[camera_key] = image.to(dtype=torch.float32) / 255.0

    processed = preprocessor(sample)
    inference_kwargs: dict[str, int] = {}
    if num_inference_steps is not None:
        inference_kwargs["num_steps"] = num_inference_steps

    started = time.perf_counter()
    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(processed, **inference_kwargs)
        action_chunk = postprocessor(action_chunk)
    latency_ms = (time.perf_counter() - started) * 1000.0

    if action_chunk.ndim != 3 or action_chunk.shape[-1] != ACTION_DIM:
        raise BehaviorLeRobotAdapterError(
            "PI0.5 inference returned an invalid action chunk shape: "
            f"{tuple(action_chunk.shape)}"
        )
    if not bool(torch.isfinite(action_chunk).all()):
        raise BehaviorLeRobotAdapterError("PI0.5 inference returned non-finite actions")

    summary = _tensor_summary(action_chunk)
    if not all(math.isfinite(summary[key]) for key in ("min", "max", "mean")):
        raise BehaviorLeRobotAdapterError("action summary contains non-finite values")
    payload = {
        "schema_version": "1.0",
        "backend": "lerobot",
        "policy_type": "pi05",
        "checkpoint": str(pretrained_dir),
        "view_dir": str(view_dir.resolve()),
        "view_stats": str(view.stats_path),
        "dataset_root": str(view.root),
        "source_revision": view.source_revision,
        "sample_index": sample_index,
        "device": device,
        "video_backend": video_backend,
        "latency_ms": latency_ms,
        "action_chunk": summary,
        "validation_status": "passed",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "inference_evidence.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"BEHAVIOR1K_PI05_INFERENCE_OK evidence={output_path}")
    print(
        "BEHAVIOR1K_PI05_ACTION "
        f"shape={summary['shape']} latency_ms={latency_ms:.2f} "
        f"min={summary['min']:.6f} max={summary['max']:.6f}"
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run real LeRobot PI0.5 checkpoint inference on one Behavior sample."
    )
    parser.add_argument("--view-dir", required=True, type=Path)
    parser.add_argument("--view-stats", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--num-inference-steps", type=int)
    args = parser.parse_args()
    run_inference(
        view_dir=args.view_dir,
        stats_path=args.view_stats,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        sample_index=args.sample_index,
        device=args.device,
        video_backend=args.video_backend,
        num_inference_steps=args.num_inference_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
