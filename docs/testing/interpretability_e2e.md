# Interpretability e2e test tier — runbook

The interpretability features (steering, clamping, capture, patching, SAE)
have three test tiers. Unit and mocked-integration tests run everywhere with
no GPU; this document is the map of the **e2e tier** — everything that needs
a real GPU, real weights, or a live server — so a validation session runs it
as one suite instead of five hand-remembered scripts.

## 0. One-time setup on the GPU box

```bash
# The dynamic-steering e2e tests resolve out-of-tree consumer stubs by
# entry point; without this install they SKIP (previously they errored).
uv pip install -e examples/capture_consumers/dynamic_steering_controller

# Point the dynamic-steering tier at a local tapped gemma4 checkpoint.
export DYNSTEER_E2E_MODEL=/path/to/gemma-4-31B-it-Q4_K_S.gguf
export DYNSTEER_E2E_LAYER=30
export VLLM_USE_FLASHINFER_SAMPLER=0   # required on some cards (see
                                       # docs/design/dynamic_steering.md §9)
```

Without `DYNSTEER_E2E_MODEL` the dynamic tests fall back to the gated
`google/gemma-4-E2B-it` with dummy weights (needs HF access to the gated
repo; unavailable access now skips instead of erroring).

## 1. Pytest e2e — offline engine, no server

Run from the repo root with the project venv.

Dynamic steering / capture→steering loop (`tests/v1/worker/`, shared
scaffolding in `tests/v1/worker/steering_e2e_utils.py`):

```bash
.venv/bin/python -m pytest -v -s \
  tests/v1/worker/test_dynamic_steering_e2e.py \
  tests/v1/worker/test_steering_gating_e2e.py \
  tests/v1/worker/test_async_steering_e2e.py \
  tests/v1/worker/test_apc_steering_e2e.py \
  tests/v1/worker/test_preemption_steering_e2e.py \
  tests/v1/worker/test_declarative_gates_e2e.py \
  tests/v1/worker/test_cross_layer_monitor_e2e.py
```

Notes:

- `test_dynamic_steering_e2e.py` and the row-gate test in
  `test_steering_gating_e2e.py` are parametrized `eager` / `cudagraph`; both
  variants must pass.
- `test_async_steering_e2e.py` and `test_cross_layer_monitor_e2e.py` need a
  *local real-weights* model (they skip under dummy weights); the cross-layer
  monitor test also needs ≥~50 layers (gemma-4-31B).
- `test_preemption_steering_e2e.py`: tune `DYNSTEER_E2E_KV_BLOCKS` (default
  200) down until `vllm:num_preemptions > 0`.

Static steering + SAE on real models
(`tests/models/language/generation/`, upstream `vllm_runner` fixture):

```bash
.venv/bin/python -m pytest -v \
  tests/models/language/generation/test_steering.py \
  tests/models/language/generation/test_steering_distributed.py \
  tests/models/language/generation/test_sae_steering_real_weights.py \
  tests/models/language/generation/test_sae_full_reconstruction_real_weights.py
```

- `test_steering.py`: dummy-weight family sweeps run on any CUDA card; the
  `*_real_weights` cases self-skip below their VRAM floors (up to 80 GiB).
- `test_steering_distributed.py`: needs 2 GPUs (full matrix 4).
- The SAE files download `google/gemma-2-2b` + a Gemma Scope SAE (gated —
  needs `HF_TOKEN`).

CUDA-gated kernel suites worth running on the same box (seconds each):

```bash
.venv/bin/python -m pytest -q \
  tests/model_executor/layers/test_sae_steering_kernel.py \
  tests/model_executor/layers/test_sae_full_reconstruction_kernel.py \
  tests/model_executor/layers/test_steering_rowgate_gpu.py
```

## 2. Manual GPU validation scripts — offline engine

Standalone scripts (not pytest-collected); each prints one `[PASS]/[FAIL]`
line per check and exits non-zero on any failure.

```bash
# Activation patching, checks A–K (alpha no-op, self-identity, cross-prompt
# replace, denoising probe, zero-ablation, mlp hooks):
uv run python tests/gpu_patch_validate.py --model Qwen/Qwen3-0.6B
uv run python tests/gpu_patch_validate.py --model Qwen/Qwen3-0.6B --enforce-eager

# Prefix-cache poisoning guard (two subprocess phases, fresh engine each):
uv run python tests/gpu_patch_poison_validate.py

# Clamp Triton-vs-eager kernel parity (no server):
uv run python tests/gpu_clamp_validate.py --mode kernel
```

## 3. Manual GPU validation scripts — live server

```bash
# Clamping over HTTP (10 checks incl. global set/clear, named modules, 400s):
vllm serve <model> --enable-steering --port 8412 &
uv run python tests/gpu_clamp_validate.py --base-url http://localhost:8412

# Server-side patch sweep vs per-cell client fan-out:
vllm serve Qwen/Qwen3-0.6B --enable-patching --port 8123 &
uv run python tests/gpu_patch_sweep_validate.py \
  --base-url http://localhost:8123/v1

# Client-provided patch vector sources (zeros/inline/module/mask/sweep/SSE):
vllm serve <model> --enable-steering --enable-patching --port 8399 &
uv run python tests/gpu_patch_vector_validate.py \
  --base-url http://localhost:8399
```

(`--enable-patching` auto-registers the `patch_source` consumer.)

## 4. Suggested validation-session order

1. `uv pip install -e examples/capture_consumers/dynamic_steering_controller`
   and export the `DYNSTEER_E2E_*` env vars (§0).
2. Kernel suites (§1, last block) — fastest signal.
3. Offline pytest e2e (§1, first two blocks).
4. Offline validation scripts (§2).
5. Live-server scripts (§3), one server at a time.
6. Record results (and any tolerance drift) in the relevant
   `docs/features/*.md` / `docs/design/*.md` GPU-validation notes.
