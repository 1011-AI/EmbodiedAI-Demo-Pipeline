#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Asset:
    profile: str
    name: str
    path: str
    kind: str = "dir_nonempty"
    required: bool = True


@dataclass
class AssetStatus:
    profile: str
    name: str
    path: str
    kind: str
    required: bool
    status: str
    detail: str


ASSETS: tuple[Asset, ...] = (
    Asset("core", "data pool", "data", "dir"),
    Asset("core", "model pool", "models", "dir"),
    Asset("core", "run pool", "runs", "dir"),
    Asset("core", "HF cache", "hf_cache", "dir"),
    Asset("core", "upstream checkout pool", "upstreams", "dir"),
    Asset("lerobot", "PushT dataset", "data/lerobot/pusht"),
    Asset("lerobot", "SO100 pick-place dataset", "data/lerobot/svla_so100_pickplace"),
    Asset("lerobot", "Diffusion PushT policy", "models/lerobot/diffusion/diffusion_pusht"),
    Asset("lerobot", "SmolVLA base policy", "models/lerobot/smolvla/smolvla_base"),
    Asset("lerobot", "pi05 base policy", "models/lerobot/pi05/pi05_base"),
    Asset("lerobot", "pi05 PaliGemma tokenizer cache", "hf_cache/hub/models--google--paligemma-3b-pt-224", required=False),
    Asset("lerobot", "FastWAM LIBERO policy", "models/lerobot/fastwam/fastwam_libero_uncond_2cam224"),
    Asset("lerobot", "FastWAM LIBERO v2.1 raw", "data/lerobot/libero-fastwam/v2.1"),
    Asset("lerobot", "FastWAM LIBERO v3 converted", "data/lerobot/libero-fastwam/v3"),
    Asset("lerobot", "Wan2.2 base cache", "hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"),
    Asset("lerobot", "UMT5 base cache", "hf_cache/hub/models--google--umt5-xxl"),
    Asset("custom-fastwam", "FastWAM native LIBERO data", "data/custom/fastwam/libero-fastwam"),
    Asset(
        "custom-fastwam",
        "FastWAM LIBERO spatial tasks",
        "data/custom/fastwam/libero-fastwam/libero_spatial_no_noops_lerobot/meta/tasks.jsonl",
        "file",
    ),
    Asset(
        "custom-fastwam",
        "FastWAM LIBERO object tasks",
        "data/custom/fastwam/libero-fastwam/libero_object_no_noops_lerobot/meta/tasks.jsonl",
        "file",
    ),
    Asset(
        "custom-fastwam",
        "FastWAM LIBERO goal tasks",
        "data/custom/fastwam/libero-fastwam/libero_goal_no_noops_lerobot/meta/tasks.jsonl",
        "file",
    ),
    Asset(
        "custom-fastwam",
        "FastWAM LIBERO 10 tasks",
        "data/custom/fastwam/libero-fastwam/libero_10_no_noops_lerobot/meta/tasks.jsonl",
        "file",
    ),
    Asset("custom-fastwam", "FastWAM release checkpoint", "models/custom/fastwam/release/libero_uncond_2cam224.pt", "file"),
    Asset("custom-fastwam", "FastWAM release stats", "models/custom/fastwam/release/libero_uncond_2cam224_dataset_stats.json", "file"),
    Asset("custom-fastwam", "FastWAM overlaid workdir", "upstreams/FastWAM-realrobot/scripts/train_zero1.sh", "file"),
    Asset("imagewam", "ImageWAM upstream", "upstreams/ImageWAM"),
    Asset("imagewam", "ImageWAM FLUX.2 4B LIBERO checkpoint", "models/custom/imagewam/flux2_klein_4b_libero"),
    Asset("imagewam", "ImageWAM FLUX.2 base", "models/custom/imagewam/flux2"),
    Asset(
        "lerobot-pi05",
        "pi05 policy config",
        "models/lerobot/pi05/pi05_base/config.json",
        "file_nonempty",
    ),
    Asset(
        "lerobot-pi05",
        "pi05 policy weights",
        "models/lerobot/pi05/pi05_base/model.safetensors",
        "file_nonempty",
    ),
    Asset(
        "lerobot-pi05",
        "PaliGemma config cache",
        "hf_cache/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c/config.json",
        "file_nonempty",
    ),
    Asset(
        "lerobot-pi05",
        "PaliGemma tokenizer config cache",
        "hf_cache/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c/tokenizer_config.json",
        "file_nonempty",
    ),
    Asset(
        "lerobot-pi05",
        "PaliGemma tokenizer cache",
        "hf_cache/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c/tokenizer.json",
        "file_nonempty",
    ),
)

PROFILE_ORDER = ("core", "lerobot", "lerobot-pi05", "custom-fastwam", "imagewam")


def _has_children(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def check_asset(root: Path, asset: Asset) -> AssetStatus:
    path = root / asset.path
    if asset.kind == "file":
        ok = path.is_file()
        status = "ok" if ok else "missing"
        detail = "file exists" if ok else "file not found"
    elif asset.kind == "file_nonempty":
        ok = path.is_file() and path.stat().st_size > 0
        status = "ok" if ok else "missing"
        detail = (
            f"file exists ({path.stat().st_size} bytes)"
            if ok
            else "non-empty file not found"
        )
    elif asset.kind == "glob_file":
        matches = sorted(item for item in root.glob(asset.path) if item.is_file())
        ok = bool(matches)
        status = "ok" if ok else "missing"
        detail = (
            f"matched {len(matches)} file(s); first={matches[0]}"
            if ok
            else "no matching file found"
        )
    elif asset.kind == "dir":
        ok = path.is_dir()
        status = "ok" if ok else "missing"
        detail = "directory exists" if ok else "directory not found"
    elif asset.kind == "dir_nonempty":
        if not path.is_dir():
            status = "missing"
            detail = "directory not found"
        elif not _has_children(path):
            status = "empty"
            detail = "directory exists but is empty"
        else:
            status = "ok"
            detail = "directory exists and is non-empty"
    else:
        raise ValueError(f"unsupported asset kind: {asset.kind}")
    return AssetStatus(
        profile=asset.profile,
        name=asset.name,
        path=str(path),
        kind=asset.kind,
        required=asset.required,
        status=status,
        detail=detail,
    )


def select_assets(profile: str) -> list[Asset]:
    if profile == "all":
        profiles = set(PROFILE_ORDER)
    else:
        profiles = {profile}
    return [asset for asset in ASSETS if asset.profile in profiles]


def print_table(results: list[AssetStatus]) -> None:
    rows = [
        (
            item.profile,
            item.status.upper(),
            item.name,
            item.path,
        )
        for item in results
    ]
    headings = ("PROFILE", "STATUS", "ASSET", "PATH")
    widths = [
        max(len(headings[index]), *(len(row[index]) for row in rows))
        for index in range(len(headings))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headings)))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check repo-local data/model/cache assets.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--profile",
        choices=("core", "lerobot", "lerobot-pi05", "custom-fastwam", "imagewam", "all"),
        default="all",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    results = [check_asset(root, asset) for asset in select_assets(args.profile)]
    if args.json:
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2))
    else:
        print_table(results)

    missing = [item for item in results if item.required and item.status != "ok"]
    if missing:
        print(
            f"\nMISSING_ASSETS profile={args.profile} count={len(missing)}. "
            "Run the matching prepare-assets-* target or see docs/BOOTSTRAP.md.",
            file=sys.stderr,
        )
        return 1
    print(f"\nASSETS_OK profile={args.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
