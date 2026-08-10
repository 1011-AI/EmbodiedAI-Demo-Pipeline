#!/usr/bin/env python3
"""Verify every Zarr chunk declared by a released Comet Orbax checkpoint.

The Hugging Face recursive tree endpoint may omit files from very large Zarr
directories.  A successful snapshot download is therefore not sufficient.
This tool derives the complete chunk grid from each leaf's ``.zarray`` and can
download only missing files from an exact repository revision.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

from huggingface_hub import hf_hub_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def leaf_names(params: Path) -> list[str]:
    metadata = json.loads((params / "_METADATA").read_text(encoding="utf-8"))
    tree = metadata.get("tree_metadata")
    if not isinstance(tree, dict) or not tree:
        raise RuntimeError(f"invalid Orbax tree metadata: {params / '_METADATA'}")
    return [".".join(ast.literal_eval(key)) for key in tree]


def expected_chunks(zarray_path: Path) -> list[str]:
    payload = json.loads(zarray_path.read_text(encoding="utf-8"))
    shape = [int(value) for value in payload["shape"]]
    chunks = [int(value) for value in payload["chunks"]]
    if len(shape) != len(chunks) or not shape:
        raise RuntimeError(f"invalid Zarr shape/chunks: {zarray_path}")
    separator = str(payload.get("dimension_separator", "."))
    axes = [range(math.ceil(size / chunk)) for size, chunk in zip(shape, chunks, strict=True)]
    return [separator.join(str(value) for value in index) for index in itertools.product(*axes)]


def download_files(
    *,
    repo_id: str,
    revision: str,
    local_dir: Path,
    filenames: list[str],
    workers: int,
) -> None:
    def download(filename: str) -> str:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=local_dir,
        )
        return filename

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download, filename): filename for filename in filenames}
        completed = 0
        for future in as_completed(futures):
            filename = futures[future]
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - network error path.
                failures.append((filename, repr(exc)))
            completed += 1
            if completed % 250 == 0 or completed == len(futures):
                print(
                    f"PI05_CHECKPOINT_DOWNLOAD completed={completed}/{len(futures)} "
                    f"failures={len(failures)}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(
            "failed downloads: "
            + json.dumps(failures[:20], ensure_ascii=False, sort_keys=True)
        )


def inspect(checkpoint: Path) -> dict[str, Any]:
    params = checkpoint / "params"
    leaves = leaf_names(params)
    missing_metadata = []
    missing_chunks = []
    expected_count = 0
    for leaf in leaves:
        directory = params / leaf
        zarray = directory / ".zarray"
        if not zarray.is_file():
            missing_metadata.append(f"params/{leaf}/.zarray")
            continue
        chunks = expected_chunks(zarray)
        expected_count += len(chunks)
        missing_chunks.extend(
            f"params/{leaf}/{chunk}"
            for chunk in chunks
            if not (directory / chunk).is_file()
        )
    return {
        "schema_version": "1.0",
        "checkpoint": str(checkpoint),
        "orbax_leaves": len(leaves),
        "expected_chunks_with_available_metadata": expected_count,
        "missing_zarray": missing_metadata,
        "missing_chunks": missing_chunks,
        "metadata_sha256": sha256(params / "_METADATA"),
        "complete": not missing_metadata and not missing_chunks,
    }


def repair(
    checkpoint: Path,
    *,
    repo_id: str,
    revision: str,
    subdir: str,
    workers: int,
) -> dict[str, Any]:
    local_dir = checkpoint.parent
    report = inspect(checkpoint)
    if report["missing_zarray"]:
        download_files(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            filenames=[f"{subdir}/{name}" for name in report["missing_zarray"]],
            workers=workers,
        )
        report = inspect(checkpoint)
    if report["missing_chunks"]:
        download_files(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            filenames=[f"{subdir}/{name}" for name in report["missing_chunks"]],
            workers=workers,
        )
        report = inspect(checkpoint)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="models/openpi_comet/pi05-b1kpt50-cs32",
        type=Path,
    )
    parser.add_argument("--repo-id", default="sunshk/openpi_comet")
    parser.add_argument(
        "--revision",
        default="61739ffbced89dd5ba1b87c30d93d6084b79b0af",
    )
    parser.add_argument("--subdir", default="pi05-b1kpt50-cs32")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    report = (
        repair(
            checkpoint,
            repo_id=args.repo_id,
            revision=args.revision,
            subdir=args.subdir,
            workers=args.workers,
        )
        if args.repair
        else inspect(checkpoint)
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
    print(encoded, end="")
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
