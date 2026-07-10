#!/usr/bin/env bash
# Build the serving venv for this fork on the CoreWeave east cluster.
#
# Installs the fork editable with VLLM_USE_PRECOMPILED (no CUDA compile; the
# steering/capture subsystems are torch-level so the upstream base wheel's
# kernels suffice). SETUPTOOLS_SCM_PRETEND_VERSION avoids needing full git
# history/tags in shallow clones.
#
# Usage:  bash examples/jlens_glm52/build-env.sh [VENV_DIR]
set -euo pipefail

VENV_DIR="${1:-/mnt/data/artifacts/jlens/vllm-env}"
FORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -x "${VENV_DIR}/bin/python" ]] && "${VENV_DIR}/bin/python" -c "import vllm" 2>/dev/null; then
  echo "[build-env] existing env at ${VENV_DIR} imports vllm; nothing to do"
  exit 0
fi

echo "[build-env] creating venv at ${VENV_DIR} (python 3.12)"
uv venv --python 3.12 "${VENV_DIR}"

export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM:-0.22.0.dev0+jlens}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM}}"
export VLLM_USE_PRECOMPILED=1

echo "[build-env] installing fork (editable, precompiled kernels) from ${FORK_DIR}"
if ! VIRTUAL_ENV="${VENV_DIR}" uv pip install -e "${FORK_DIR}" --python "${VENV_DIR}/bin/python" 2>&1 | tail -5; then
  echo "[build-env] PRECOMPILED INSTALL FAILED (exit 42)." >&2
  echo "[build-env] Fallback is a source build: CUDA_HOME=/mnt/data/artifacts/cuda-12.9 \\" >&2
  echo "[build-env]   VLLM_USE_PRECOMPILED=0 uv pip install -e ${FORK_DIR} (slow, ~1h)." >&2
  exit 42
fi

"${VENV_DIR}/bin/python" -c "import vllm; print('[build-env] vllm', vllm.__version__, 'OK')"
# openai client for the demo scripts; transformers ships with vllm.
VIRTUAL_ENV="${VENV_DIR}" uv pip install --python "${VENV_DIR}/bin/python" openai >/dev/null
echo "[build-env] done: ${VENV_DIR}"
