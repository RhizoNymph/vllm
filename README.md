<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM with activation steering and activation capture
</h3>

---

## About This Fork

This is a fork of [vLLM](https://github.com/vllm-project/vllm) that adds two
interpretability subsystems wired directly into the model forward pass, built to
the same production bar as the rest of the engine:

- **Activation steering** — add precomputed vectors into the residual stream at
  inference time to shift model behavior without fine-tuning (tone/style,
  behavioral interventions, SAE-derived steering).
- **Activation capture** — a pluggable consumer system that routes hidden-state
  activations out of the forward pass to disk, a training loop, a dashboard, or
  any third-party plugin. Ships with a built-in filesystem consumer.

Both hook the residual stream at three points — `pre_attn`, `post_attn`,
`post_block` — across 100+ decoder architectures, and both run under continuous
batching, `torch.compile`, CUDA graphs, and tensor/pipeline parallelism.

📖 **Full guides:** [Activation Steering](docs/features/steering.md) ·
[Activation Capture](docs/features/capture_consumers.md)

## Design highlights

The hard part isn't adding a vector to the residual stream — it's doing so
without giving up vLLM's performance and correctness guarantees. The notable
engineering:

- **Stays on the CUDA-graph fast path.** Naively, capturing per-request
  activations forces every step eager. Instead, a rank-replicated per-step gate
  runs eager *only* on steps that actually gather, global probes use a
  persistent-buffer `copy_` baked into the graph at warmup, and a startup
  allowlist (`--capture-graphsafe-key`) lets chosen per-request keys ride that
  same graph-safe path. Plain traffic on a capture-enabled server keeps full
  cudagraph speed.
- **Prefix-cache correct.** Steering forks the APC cache key on prefill steering
  (but not decode-only steering, which must not fork prompt KV); capture
  re-forwards only the prompt suffix it needs when a tapped position was served
  from cache. Steering correctness under APC is treated as a correctness
  requirement, not a perf nicety.
- **Distributed.** Global steering fans out to every worker via `collective_rpc`
  with lock-step row allocation and no hot-path coordination; capture merges
  per-stage results under pipeline parallelism. Both validated across TP, PP, and
  cross-node.
- **Pluggable + tuned for real storage.** Capture consumers are entry-point
  plugins; the filesystem consumer offers `per_file` / `packed` / `sharded`
  layouts because on a network mount throughput is governed by file count
  (metadata RPCs), not bytes.

## Performance

Everything below is measured against an unmodified baseline (same build,
features disabled). Methodology and profiling deep-dives are in the two
write-ups: [Activation Steering in vLLM](https://www.rhizonymph.com/blog/activation-steering/)
· [Part 2](https://www.rhizonymph.com/blog/activation-steering-boogaloo/).

### Steering overhead

Gemma-3-27B on A100 — 64–128 prompts, concurrency 8–16, `max_tokens=256`,
vs. disabled baseline (TTFT 111 ms, TPOT 37.4 ms):

| Mode | ΔTTFT | ΔTPOT | ΔE2E latency |
|---|---|---|---|
| Enabled, no active configs | −6.6 ms | ±0.0% | −0.1% |
| Named (pre-registered) vectors, all requests steered | +0.1 ms | +0.0% | +0.0% |
| Inline shared vectors, all requests steered | +50.5 ms | +0.5% | +1.7% |
| **Worst case: 16 distinct inline per-request configs** | +70.7 ms | **+1.3%** | **+2.7%** |

- CUDA graphs stay intact in every mode — 6.1–6.8× over eager on the same
  workload — and per-token cost is flat: the only real cost is per-request
  submission, which amortizes with output length (worst case on the 4B model:
  +17.3% E2EL at 64 output tokens → +1.7% at 2048).
- Memory: ~522 KB per steering config on the 4B model, <0.15% of weights VRAM.
- The worst case started at **+22% E2EL**. Getting to +2.7% took: a binary wire
  format for inline vectors (TTFT −77–79%, throughput +22–30% — the bottleneck
  was ~87k Python floats per request materialized in the server event loop),
  worker-side named-vector resolution (~599 KB → 214 bytes per request),
  batched registration (79.6 → 15.6 ms per request), and a fused Triton
  gather-add kernel (~40% less HBM traffic at the op).

### Capture overhead and filesystem throughput

Fixed-clock A/B runs, Gemma-3 on RTX 3090s, NFS over a 20 GbE bond:

- Non-capturing traffic on a capture-enabled server keeps full CUDA-graph
  speed: eager is forced only on steps that actually gather (replacing a
  blanket always-eager cost of +14% TPOT).
- Per-request (non-global) capture is performant too, via three graph-safety
  tiers measured on a dense per-request capture workload: server-global specs
  on the persistent-buffer path cut the decode-step penalty **+287% → +11%**;
  the `--capture-graphsafe-key` allowlist takes per-request taps from
  **+400% → +14%** (sparse capture flat); and on the v2 runner,
  piecewise-graph fallback caps the worst no-allowlist case at ~+107%
  instead of +293%.
- End-to-end cost for requests that *do* capture: **−2% throughput**, down from
  −23% before step-gating and non-blocking finalize; capture-request TTFT
  1391 → 877 ms.
- Filesystem consumer throughput is governed by file count (metadata RPCs),
  not bytes — measured at 32 requests × 24 layers, fp32:

| Layout | Files written | Throughput | Finalize p50 |
|---|---|---|---|
| `per_file` | 768 | 29 MB/s | 2.26 s |
| `packed` | 32 | 142 MB/s | 469 ms |
| `sharded` | 8 | 505 MB/s | 6.6 ms |

Large-file capture sustains 331–372 MB/s to the NFS mount — ~93% of its
measured 398 MB/s disk bound.

### Prefix caching preserved

Steering keeps automatic prefix caching intact rather than disabling it
(gemma-3-4b-it, RTX 3090, ~1500-token shared prefix, concurrency 24):
unsteered, shared-config, and decode-only-steered traffic all hold a 98%
cache hit rate and the full ~4× TTFT / ~3–3.8× throughput benefit of APC.
Decode-only steering never forks prompt KV by construction. A distinct
prefill config per request drops the hit rate to 23% — the cache key forks
because the KV genuinely differs under different steering; that fork is the
correctness contract, and outputs were verified deterministic across cache
regimes.

### Dynamic steering (in-progress branch)

The activation-conditioned steering stack (see Roadmap) is benchmarked on its
own branch — gemma-3-4b, RTX 3090 with locked clocks, CUDA graphs on, overhead
flat across batch 1–32:

| Configuration | Overhead vs own baseline |
|---|---|
| Steering compiled in, idle | 0% |
| Global-tier dynamic (in-graph gate) | ~+1.5% |
| Per-request dynamic (override pool) | ~+2.3% |
| Static per-request steering | ~+2.7% |
| Async transport (capture → D2H → dispatch pipeline) | +5–6.7% |

- The ordering is the point: the dynamic paths cost *less* than static
  per-request steering, and the in-graph monitor is effectively free — it
  reads persistent buffers inside the replayed graph. Async is the outlier
  because it pays for the full capture-gather/D2H/dispatch pipeline; sync and
  in-graph paths don't.
- Sync-consumer actuation on a 31B model: **~0.05 ms/step** (~0.16% of a 31 ms
  decode step) by CUDA-event measurement; serving A/B within noise. The
  one-time ~117 ms probe/cuBLAS init is pre-paid by a warmup hook.
- The residual cost is a roughly fixed ~8 ms/step of host work: ~3–5% on a
  small host-bound model, approaching 0% on large GPU-bound models.
- An earlier apparent +8.6% CUDA-graph regression at decode batch >16 was
  root-caused as a node-specific measurement confound — nsys showed
  byte-identical GPU kernel time across arms, and it did not reproduce on
  other nodes, models, or the merged branch under locked clocks.

## Quickstart

**Steering** — global state over HTTP, or per-request via `SamplingParams`:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="google/gemma-3-4b-it", enable_steering=True)
params = SamplingParams(
    max_tokens=64,
    steering_vectors={"post_block": {15: {"vector": [0.1, 0.2], "scale": 2.0}}},
    decode_steering_vectors={"pre_attn": {15: [0.5, 0.6]}},
)
outputs = llm.generate(["Hello"], params)
```

Steering composes three additive tiers (base / prefill / decode) at both the
global and per-request level, supports named pre-registered modules, and exposes
`/v1/steering/*` endpoints (gated by `VLLM_SERVER_DEV_MODE=1`). See the
[steering guide](docs/features/steering.md) and the runnable
[`examples/online_serving/openai_steering_client.py`](examples/online_serving/openai_steering_client.py).

**Capture** — enable a consumer, then opt requests in:

```bash
vllm serve meta-llama/Llama-3-8B \
    --capture-consumers filesystem:root=/mnt/nas/activations
```

```python
from vllm import SamplingParams
from vllm.v1.capture.consumers.filesystem import FilesystemCaptureRequest

params = SamplingParams(
    max_tokens=16,
    capture={"filesystem": FilesystemCaptureRequest(
        request_id="probe_0001", tag="mnist-probe-v1",
        hooks={"post_block": [12]}, positions="last_prompt",
    )},
)
```

Consumers can be global (every request, e.g. a `logging` probe) or per-request
(client-driven). Results come back on `RequestOutput.capture_results`. See the
[capture guide](docs/features/capture_consumers.md) for layouts, throughput
tuning, backpressure policies, and the plugin-authoring path, plus example
plugins under [`examples/capture_consumers/`](examples/capture_consumers/).

## Supported models

Hooks are wired into the Llama, Qwen, Gemma, Mixtral/MoE, GLM, InternLM, Olmo,
Exaone, Phi, Plamo, Step, Molmo, Falcon/Baichuan/Command/StableLM families and
more — 100+ decoder architectures. Gemma 3 is the primary end-to-end test
target; several MoE models are validated against real weights. See the full
list in the [steering guide](docs/features/steering.md#supported-scope).

## Roadmap

> In progress / planned; details **subject to change**.

- **Dynamic steering** *(next; open draft [PR #180](https://github.com/RhizoNymph/vllm/pull/180))* —
  activation-conditioned steering that ties capture to steering so the model's
  own activations decide *when* and *how* to steer. A stack of three controller
  tiers — async (steers a later request), sync (per-step, every TP rank), and an
  in-graph monitor that gates a dynamic steering tier *within the same forward
  pass* via `sigmoid(sharpness · (residual · probe − threshold))` — plus the
  APC-correctness notification so dynamically-steered decode KV isn't falsely
  reused. GPU-validated on gemma4-31B across TP=1, TP=2 (cross-node), and PP=2;
  overhead measured under CUDA graphs — the dynamic paths cost less than static
  per-request steering (see [Performance](#performance)).
- **Activation patching at scale** *(after that)* — transplanting/overwriting
  captured activations across runs as a first-class, high-throughput operation.
  Direction marker; details not yet pinned down.

## Upstream vLLM

This README covers only the steering/capture additions. For installation,
supported-model details, the OpenAI-compatible server, quantization, distributed
inference, and all other vLLM functionality, see the upstream project —
installation is unchanged from upstream:

- Repository: <https://github.com/vllm-project/vllm>
- Documentation: <https://docs.vllm.ai>

## Citation

If you use vLLM for your research, please cite the
[paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```
