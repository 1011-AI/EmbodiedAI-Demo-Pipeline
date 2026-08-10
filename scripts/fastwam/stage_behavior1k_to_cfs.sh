#!/usr/bin/env bash
set -euo pipefail

# Resumable, non-destructive staging of the assets used by FastWAM training.
# The BOS source is never modified.  Set BEHAVIOR1K_STAGE_SCOPE=full to include
# depth videos after the RGB training set is ready.
source_root="${BEHAVIOR1K_BOS_ROOT:-/mnt/bos/bos_0/datasets/2026-challenge-demos}"
target_root="${BEHAVIOR1K_CFS_ROOT:-/mnt/cfs/data_file_0/datasets/2026-challenge-demos}"
scope="${BEHAVIOR1K_STAGE_SCOPE:-training}"
jobs="${BEHAVIOR1K_STAGE_JOBS:-4}"

if [[ ! -f "${source_root}/meta/info.json" ]]; then
  echo "ERROR: invalid BEHAVIOR1K source: ${source_root}" >&2
  exit 2
fi
if [[ "${scope}" != "training" && "${scope}" != "full" ]]; then
  echo "ERROR: BEHAVIOR1K_STAGE_SCOPE must be training or full" >&2
  exit 2
fi

mkdir -p "${target_root}"
assets=(meta annotations data)
if [[ "${scope}" == "full" ]]; then
  assets+=(videos)
else
  assets+=(
    videos/observation.rgb.left_realsense_link_camera_0
    videos/observation.rgb.right_realsense_link_camera_0
    videos/observation.rgb.zed_link_camera_0
  )
fi

copy_asset() {
  local relative="$1"
  local source_path="${source_root}/${relative}"
  local target_parent="${target_root}/$(dirname "${relative}")"
  mkdir -p "${target_parent}"
  rsync -a --partial --partial-dir=.rsync-partial \
    "${source_path}" "${target_parent}/"
}

export source_root target_root
export -f copy_asset
printf '%s\n' "${assets[@]}" | xargs -P "${jobs}" -I '{}' bash -c 'copy_asset "$1"' _ '{}'

for relative in "${assets[@]}"; do
  source_count="$(find "${source_root}/${relative}" -type f | wc -l)"
  target_count="$(find "${target_root}/${relative}" -type f ! -path '*/.rsync-partial/*' | wc -l)"
  if [[ "${source_count}" != "${target_count}" ]]; then
    echo "ERROR: file-count mismatch for ${relative}: ${source_count} != ${target_count}" >&2
    exit 3
  fi
  source_bytes="$(find "${source_root}/${relative}" -type f -printf '%s\n' | awk '{total += $1} END {printf "%.0f", total}')"
  target_bytes="$(find "${target_root}/${relative}" -type f ! -path '*/.rsync-partial/*' -printf '%s\n' | awk '{total += $1} END {printf "%.0f", total}')"
  if [[ "${source_bytes}" != "${target_bytes}" ]]; then
    echo "ERROR: byte-count mismatch for ${relative}: ${source_bytes} != ${target_bytes}" >&2
    exit 3
  fi
  echo "VERIFIED ${relative} files=${source_count} bytes=${source_bytes}"
done

marker="${target_root}/.fastwam_training_assets_ready"
{
  echo "source=${source_root}"
  echo "scope=${scope}"
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${marker}"
echo "BEHAVIOR1K_CFS_STAGE_READY ${target_root} scope=${scope}"
