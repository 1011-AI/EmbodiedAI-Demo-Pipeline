"""Read-only integrity checks for prepared PI0.5-Comet backends."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


MANIFEST_PATH = Path("overlays/pi05_comet/prepared_backend_manifest.json")
TREE_HASH_ALGORITHM = "sha256-path-length-path-size-content-v1"


class BackendIntegrityError(RuntimeError):
    """Raised when a prepared backend differs from its immutable manifest."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def python_tree_sha256(root: Path) -> tuple[str, int]:
    """Hash Python source paths and contents without depending on Git metadata."""

    if not root.is_dir():
        raise BackendIntegrityError(f"prepared source tree is missing: {root}")
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        raise BackendIntegrityError(
            f"prepared source tree contains no Python files: {root}"
        )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), len(files)


def _mapping(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise BackendIntegrityError(
            f"prepared backend manifest field {name!r} must be an object"
        )
    return dict(value)


def _project_path(project_root: Path, relative: Any) -> Path:
    value = Path(str(relative))
    if value.is_absolute():
        raise BackendIntegrityError(
            f"prepared backend manifest path must be relative: {value}"
        )
    resolved = (project_root / value).resolve()
    if not resolved.is_relative_to(project_root):
        raise BackendIntegrityError(f"prepared backend manifest path escapes project: {value}")
    return resolved


def verify_prepared_backends(
    project_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify shared-disk backend sources without running Git or changing files."""

    root = Path(project_root).resolve()
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else (root / MANIFEST_PATH).resolve()
    )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendIntegrityError(
            f"cannot read prepared backend manifest {manifest}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise BackendIntegrityError(f"unsupported prepared backend manifest: {manifest}")
    if payload.get("tree_hash_algorithm") != TREE_HASH_ALGORITHM:
        raise BackendIntegrityError(
            f"prepared backend tree hash algorithm mismatch: {payload.get('tree_hash_algorithm')!r}"
        )

    report: dict[str, Any] = {
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "tree_hash_algorithm": TREE_HASH_ALGORITHM,
        "backends": {},
        "artifacts": {},
    }
    for name in ("openpi_comet", "lerobot"):
        expected = _mapping(payload, name)
        revision = str(expected.get("revision", ""))
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise BackendIntegrityError(
                f"invalid prepared revision for {name}: {revision!r}"
            )
        source_root = _project_path(root, expected.get("source_root", ""))
        actual_sha256, actual_files = python_tree_sha256(source_root)
        expected_sha256 = str(expected.get("python_tree_sha256", ""))
        expected_files = int(expected.get("python_files", -1))
        if actual_sha256 != expected_sha256 or actual_files != expected_files:
            raise BackendIntegrityError(
                f"prepared {name} source mismatch: "
                f"sha256={actual_sha256} files={actual_files}, "
                f"expected_sha256={expected_sha256} expected_files={expected_files}"
            )
        report["backends"][name] = {
            "revision": revision,
            "source_root": str(source_root),
            "python_tree_sha256": actual_sha256,
            "python_files": actual_files,
        }

    artifacts = _mapping(payload, "artifacts")
    for relative, expected_sha256 in sorted(artifacts.items()):
        path = _project_path(root, relative)
        if not path.is_file():
            raise BackendIntegrityError(f"prepared backend artifact is missing: {path}")
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise BackendIntegrityError(
                f"prepared backend artifact mismatch: {path} "
                f"sha256={actual_sha256} expected={expected_sha256}"
            )
        report["artifacts"][relative] = actual_sha256
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify_prepared_backends(args.project_root, args.manifest)
    except BackendIntegrityError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print("PI05_PREPARED_BACKENDS_OK " + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
