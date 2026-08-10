#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "${PYTHON_BIN} not found; read docs/ENVIRONMENT.md"

"${PYTHON_BIN}" -c '
import os
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ is required, found {sys.version.split()[0]}")
if os.environ.get("VIRTUAL_ENV") or sys.prefix != sys.base_prefix:
    raise SystemExit("a virtual environment is active; use the image default Python")
print(f"python={sys.executable}")
print(f"python_version={sys.version.split()[0]}")
'

"${PYTHON_BIN}" -c '
import platform
print(f"platform={platform.system()}-{platform.machine()}")
'

if ! "${PYTHON_BIN}" -m pip check; then
  printf 'WARNING: pip metadata has known upstream pins; runtime imports remain the training-image contract.\n' >&2
fi
"${PYTHON_BIN}" -m embodied_demo --version
"${PYTHON_BIN}" -c '
import av
import deepspeed
import diffusers
import fastwam
import pyarrow
import torch
import torchcodec

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"visible_gpus={torch.cuda.device_count()}")
print(f"deepspeed={deepspeed.__version__}")
print(f"diffusers={diffusers.__version__}")
print(f"pyarrow={pyarrow.__version__}")
print(f"pyav={av.__version__}")
print(f"torchcodec={torchcodec.__version__}")
print(f"fastwam={fastwam.__file__}")
'

if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -version | head -1
else
  fail "ffmpeg not found; TorchCodec requires system FFmpeg shared libraries"
fi

if command -v gh >/dev/null 2>&1; then
  printf 'gh=%s\n' "$(gh --version | head -1)"
else
  printf 'INFO: gh is optional for runtime and required only for GitHub publishing.\n'
fi

if [[ -n "${HTTPS_PROXY:-${https_proxy:-}}" ]]; then
  printf 'shell_proxy=configured\n'
else
  printf 'INFO: shell proxy is not configured; this is valid on direct-connect networks.\n'
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia_smi=available\n'
else
  printf 'INFO: NVIDIA runtime not detected; core checks can pass, but training requires a GPU node.\n'
fi

printf 'environment_status=OK\n'
