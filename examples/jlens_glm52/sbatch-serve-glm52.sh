#!/usr/bin/env bash
#SBATCH --job-name=jlens-serve-glm52
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Serve GLM-5.2 on this fork: single B200 node, TP8, on-the-fly FP8 weight
# quantization (bf16 checkpoint is 1.4 TB and does not fit one node), with
# steering enabled and the filesystem capture consumer mounted.
#
# The server hostname:port is written to ${RUN_INFO} (default
# /mnt/data/artifacts/jlens/serve/current.json) for the demo clients.
#
# Env knobs:
#   MODEL            HF dir (default /mnt/data/artifacts/GLM-5.2)
#   VENV_DIR         serving venv (default /mnt/data/artifacts/jlens/vllm-env;
#                    built on the fly by build-env.sh if missing)
#   PORT             default 8000
#   MAX_MODEL_LEN    default 8192
#   QUANTIZATION     default fp8 ("" -> bf16; needs 2 nodes, not this script)
#   CAPTURE_ROOT     default /mnt/data/artifacts/jlens/vllm_capture
#   GRAPHSAFE_KEYS   default "30:post_block 40:post_block 50:post_block"
#   EXTRA_ARGS       appended verbatim to vllm serve
#
# Usage: sbatch examples/jlens_glm52/sbatch-serve-glm52.sh
set -euo pipefail

MODEL="${MODEL:-/mnt/data/artifacts/GLM-5.2}"
VENV_DIR="${VENV_DIR:-/mnt/data/artifacts/jlens/vllm-env}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
QUANTIZATION="${QUANTIZATION:-fp8}"
CAPTURE_ROOT="${CAPTURE_ROOT:-/mnt/data/artifacts/jlens/vllm_capture}"
# lens capture layers: stride-4 depth sweep + final (must match the jlens
# consumer and the sidecar's JLENS_CAPTURE_LAYERS)
JLENS_LAYERS="${JLENS_LAYERS:-$(seq -s';' 0 4 76);77}"
GRAPHSAFE_KEYS="${GRAPHSAFE_KEYS:-$(echo "${JLENS_LAYERS}" | tr ';' '\n' | sed 's/$/:post_block/' | tr '\n' ' ')}"
RUN_INFO="${RUN_INFO:-/mnt/data/artifacts/jlens/serve/current.json}"

FORK_DIR="${SLURM_SUBMIT_DIR:?submit from the fork repo root}"
[[ -f "${FORK_DIR}/examples/jlens_glm52/build-env.sh" ]] || {
  echo "error: submit from the vllm fork root" >&2; exit 1; }

bash "${FORK_DIR}/examples/jlens_glm52/build-env.sh" "${VENV_DIR}"

mkdir -p "${CAPTURE_ROOT}" "$(dirname "${RUN_INFO}")"
HOST="$(hostname -f)"
cat > "${RUN_INFO}" <<EOF
{"host": "${HOST}", "port": ${PORT}, "job": "${SLURM_JOB_ID}",
 "model": "glm-5.2", "quantization": "${QUANTIZATION}"}
EOF
echo "[serve] ${HOST}:${PORT} (job ${SLURM_JOB_ID}) -> ${RUN_INFO}"

GRAPHSAFE_ARGS=()
for key in ${GRAPHSAFE_KEYS}; do
  GRAPHSAFE_ARGS+=(--capture-graphsafe-key "${key}")
done
QUANT_ARGS=()
[[ -n "${QUANTIZATION}" ]] && QUANT_ARGS=(--quantization "${QUANTIZATION}")

export VLLM_SERVER_DEV_MODE=1  # mounts /v1/steering/* inspection endpoints
source "${VENV_DIR}/bin/activate"

# Optional lens-console UI sidecar (examples/jlens_glm52/ui/) on this node —
# same node as the jlens consumer so its JSONL stream is visible without NFS
# attribute-cache lag. Expose with: tunnel-url ${SIDECAR_PORT}
if [[ "${SIDECAR:-1}" == "1" ]]; then
  SIDECAR_PORT="${SIDECAR_PORT:-7860}"
  python "${FORK_DIR}/examples/jlens_glm52/ui/sidecar.py" --port "${SIDECAR_PORT}" \
    > "$(dirname "${RUN_INFO}")/sidecar-${SLURM_JOB_ID}.log" 2>&1 &
  echo "[serve] jlens console sidecar on :${SIDECAR_PORT} (pid $!)"
fi

exec vllm serve "${MODEL}" \
  --served-model-name glm-5.2 \
  --tensor-parallel-size 8 \
  "${QUANT_ARGS[@]}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --port "${PORT}" \
  --enable-steering \
  --capture-consumers "filesystem:root=${CAPTURE_ROOT}" \
  --capture-consumers "jlens:lens=${JLENS_LENS:-/mnt/data/artifacts/jlens/glm52_fit_1k/lens_glm52_1k.pt},unembed=${JLENS_UNEMBED:-/mnt/data/artifacts/jlens/glm52_unembed.pt},layers=${JLENS_LAYERS},topk=8,out=${JLENS_READOUT:-/mnt/data/artifacts/jlens/readout},device=cuda:0" \
  "${GRAPHSAFE_ARGS[@]}" \
  --trust-remote-code \
  ${EXTRA_ARGS:-}
