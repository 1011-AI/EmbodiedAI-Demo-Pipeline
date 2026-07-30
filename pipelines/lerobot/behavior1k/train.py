"""Real LeRobot PI0.5 training entry with the BEHAVIOR-1K dataset hook.

Example (single process):

    python -m pipelines.lerobot.behavior1k.train \
      --behavior-view-dir data/behavior1k/views/r1pro_policy23/turning_on_radio \
      --policy.type=pi05 ...

The same module can be launched by ``accelerate launch --module``.  All
arguments except the two ``--behavior-*`` options are passed unchanged to the
pinned LeRobot ``lerobot_train`` parser.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from .adapter import (
    BehaviorLeRobotAdapterError,
    load_behavior_view,
    make_behavior_train_eval_datasets,
)
from .loading import (
    direct_cuda_load_requested,
    make_policy_with_memory_strategy,
)


def _extract_behavior_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--behavior-view-dir",
        default=os.environ.get("BEHAVIOR1K_VIEW_DIR"),
    )
    parser.add_argument(
        "--behavior-view-stats",
        default=os.environ.get("BEHAVIOR1K_VIEW_STATS"),
    )
    args, remaining = parser.parse_known_args(list(argv))
    if not args.behavior_view_dir:
        raise BehaviorLeRobotAdapterError(
            "set --behavior-view-dir or BEHAVIOR1K_VIEW_DIR"
        )
    return args, remaining


def main(argv: Sequence[str] | None = None) -> None:
    behavior_args, lerobot_args = _extract_behavior_args(
        sys.argv[1:] if argv is None else argv
    )
    view = load_behavior_view(
        behavior_args.behavior_view_dir,
        stats_path=behavior_args.behavior_view_stats,
    )

    try:
        from lerobot.datasets.factory import (
            make_train_eval_datasets as upstream_factory,
        )
        from lerobot.scripts import lerobot_train
    except ImportError as exc:
        raise BehaviorLeRobotAdapterError(
            "the pinned LeRobot training package is not importable"
        ) from exc
    if not hasattr(lerobot_train, "make_train_eval_datasets"):
        raise BehaviorLeRobotAdapterError(
            "unsupported LeRobot API: lerobot_train has no module-level dataset factory"
        )
    original_make_policy = getattr(lerobot_train, "make_policy", None)
    if original_make_policy is None and direct_cuda_load_requested():
        raise BehaviorLeRobotAdapterError(
            "direct CUDA loading requires lerobot_train.make_policy"
        )

    original_factory = lerobot_train.make_train_eval_datasets

    def behavior_factory(cfg):
        return make_behavior_train_eval_datasets(
            cfg,
            view=view,
            upstream_factory=upstream_factory,
        )

    def behavior_make_policy(*, cfg, ds_meta=None, env_cfg=None, rename_map=None):
        return make_policy_with_memory_strategy(
            original_make_policy,
            cfg=cfg,
            ds_meta=ds_meta,
            env_cfg=env_cfg,
            rename_map=rename_map,
        )

    lerobot_train.make_train_eval_datasets = behavior_factory
    if original_make_policy is not None:
        lerobot_train.make_policy = behavior_make_policy
    original_argv = sys.argv
    sys.argv = [original_argv[0], *lerobot_args]
    try:
        lerobot_train.main()
    finally:
        sys.argv = original_argv
        lerobot_train.make_train_eval_datasets = original_factory
        if original_make_policy is not None:
            lerobot_train.make_policy = original_make_policy


if __name__ == "__main__":
    main()
