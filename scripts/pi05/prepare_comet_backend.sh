#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Runtime preparation is intentionally read-only. The overlay and both source
# checkouts are staged on shared storage before a Baige allocation; every node
# verifies the immutable source manifest without requiring a Git executable or
# racing to modify the shared checkout.
cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}" \
  python -m embodied_demo.pi05_backend_integrity --project-root "${PROJECT_ROOT}"
