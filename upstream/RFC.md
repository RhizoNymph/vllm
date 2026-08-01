# RFC - Activation Steering Runtime and Wiring

## Summary
This RFC proposes adding activation steering to vLLM with a GPU-resident steering
runtime, an API surface for per-request and global steering, and the model
wiring to expose it on supported decoders (starting with Gemma 3). The runtime
is built around a custom opaque torch op and per-layer steering tables with
deterministic row assignment, preserving torch.compile capture, prefix-cache
correctness, and tensor/pipeline parallel equivalence without NCCL on the hot
path. Users configure steering per-request via SamplingParams (or OpenAI
extra_body) and, optionally, globally via a gated HTTP API with named modules.
Overhead is within noise when enabled-but-idle and amortizes to single-digit
percentages on long outputs; details in the Performance section. The runtime is
also proposed as the substrate for Phase 2 dynamic steering in
https://github.com/vllm-project/vllm/issues/36998.

This is not a duplicate of the prior `ControlVectorRequest` attempts
(#5807, #7906, #12870); those are scoped to per-request vectors only
and their self-reported defects are addressed by the current design.
Full audit in Prior Art below.

## Motivation
Activation steering is a well-established research direction for
mechanistic interpretability, but the open-source community does not
currently have an inference solution that avoids sacrificing throughput
to enable it. As well as speeding up research workflows, an in-tree
runtime would let that research be applied in production, which cannot
happen unless a production-grade inference server integrates
activation steering. The fork this work was developed on is currently
in use by a mechanistic interpretability researcher who has reported
that it has sped up her workflow by 1–2 orders of magnitude.

The class of methods this runtime serves is well-established:
Activation Addition (Turner et al. 2023), Representation Engineering
(Zou et al. 2023), Contrastive Activation Addition (Panickssery et al.
2023), Inference-Time Intervention (Li et al. 2023), and SAE-feature
steering (Templeton/Anthropic 2024; Lieberum/DeepMind 2024, "Gemma
Scope"). Gemma Scope 2 in particular ships open-weight SAEs for
Gemma-family architectures, directly usable as steering-vector sources
for the model wired up in PR 1.

Community demand has outlasted the upstream review pipeline: after the
in-tree attempts stalled, the ZJU-REAL group continued the work as the
out-of-tree "EasySteer" fork (see Prior Art). An in-tree runtime
removes the fork-replacement tax that currently blocks production
adoption.

## Design
### Goals
The goals of the design are:
- preserve torch.compile compatibility with no CUDA graph breaks
- ensure prefix cache correctness,
- integrate steering smoothly into continuous batching.

The steering functionality it implements is:
- activation steering which adds specified vectors to the residual stream at chosen hook points
- both global and per-request steering, with global steering optionally gated behind a steering operator api key
- phase-aware steering that allows specifying prefill/decode vectors separately
- distributed inference support for both tensor and pipeline parallelism

### Runtime
- Per-layer buffers on the GPU of steering_table_{hook} and steering index. Row 0 is the zero vector sentinel so that unsteered
requests can always have a no-op vector to add, and rows 1/2 are global prefill/decode vectors that are which are composed additively with per-request configs at registration time. Very low memory cost.
- SteeringManager class per worker owns the row allocation process using a deterministic config to row assignment
that uses refcounting to avoid duplication
- Custom opaque op 'torch.ops.vllm.apply_steering' performs an indexed gather from steering tables to the residual
stream based on the per-token steering index
- A shared nothing deterministic row assignment contract allows every rank in TP inference to store their own copy
of steering tables but only materializes the tensors required for PP rank workers. This avoids requiring NCCL on the steering
hot path and is preferable due to the low memory burden
- Adds steering config hashes to the cache key for the prefix cache, but must invalidate the KV cache on a global steering config change (which should be acceptable as a relatively rare event)
- Uses a two stage queue model for pending registrations and pending prefill to decode transitions (transitions take priority for UX and minimizing allocated resources) when steering tables are full.

Much of the implementation surface was intentionally modeled after the LoRA code.

![Steering architecture diagram](https://www.rhizonymph.com/blog-posts/activation-steering/steering-diagram.webp)

### API
#### SamplingParams/extra_body
- Adds steering_vectors, prefill_steering_vectors, and decode_steering_vectors fields to SamplingParams. All three operate on
json configs of { <hook_point>: { <layer_N>: [steering vector elements] } }
- For OpenAI API, configs are specified in extra_body under the same keys and format as in SamplingParams

#### CLI flags
- `--enable-steering`
- `--steering-api-key`
- `--max-steering-configs`
- `--steering-modules`

#### Hook points
- pre-attn
- post-attn
- post-mlp

#### Named modules
Allows defining name steering configs as modules to improve UX and decrease size of requests.

## Performance
### Methodology
- Hardware: H100 NVL 94GB
- Model: Gemma 3 at 4B and 27B
- Workload generator: https://github.com/RhizoNymph/steering-bench (scripts/run_h100.sh) - uses ShareGPT for realistic workloads
- vLLM commit: acf39ab1a (short-session) / 5f2737865 (long-gen session)
- Python 3.12.13, PyTorch 2.10.0+cu128 / +cu130, CUDA 12.8 / 13.0
- GPU clocks pinned at 1785 MHz for latency runs; free-running for
the 27B serving runs (vast.ai constraint)
- Prefix caching: enabled; max_steering_configs: 32 (matrix) / 16 (serving)
- Warmup 10 iters, measured 30 iters per point
- Configurations:
  - disabled: no --enable-steering used
  - enabled_idle: --enable-steering but all requests unsteered
  - all_steered_shared: 16 batch size, all requests in batch are steered but share two active configs
  - per_request_n4: 16 batch size, 4 steered requests each with distinct configs
  - per_request_n16: 16 batch size, all requests steered with distinct configs

### Claims and numbers
- Zero overhead when enabled but idle

  enabled_idle is within run-to-run noise of disabled across every metric and
  configuration tested — turning the feature on is free until a request actually
  asks for steering.

| Metric | 4B worst delta | 27B worst delta |
| :--- | :--- | :--- |
| TPOT (med) | +2.0% @ 256 tok | +0.4% @ 2048 tok |
| TTFT (med) | +10% @ 2048 tok | 0% |
| E2EL @ 2048 tok | +0% | +2% |

- Time per output token amortizes to low single digit percentages on large models/long outputs

| max_tokens | 4B | 27B |
| :--- | :--- | :--- |
| 256 | +35.3% | +9.3% |
| 1024 | +15.1% | +9.4% |
| 2048 | +13.5% | +4.8% |

- Cost amortizes with sequence length

Worst-case workload (4B, all requests steered, all distinct configs):

| max_tokens | disabled tok/s | steered tok/s | % of baseline |
| :--- | :--- | :--- | :--- |
| 64 | 3590 | 1632 | 45% |
| 128 | 3670 | 2264 | 61% |
| 256 | 3638 | 2776 | 76% |
| 1024 | 3378 | 3134 | 93% |
| 2048 | 3200 | 3076 | 96% |

  At short-output / high-QPS workloads the fixed per-request registration cost
  dominates; by 1024+ tokens it's in the noise.

- Per request cost scales with active steered requests, not batch size

  4B, sweeping total batch size while holding active steered requests constant at
  N=16 distinct configs:

| Batch size | Per-active-request cost (ms) |
| :--- | :--- |
| 64 | 30.9 |
| 128 | 30.9 |
| 256 | 31.0 |
| 384 | 31.6 |

- Time to first token is where cost concentrates

  Worst-case configurations hit TTFT hard because per-request registration happens
  in prefill:

| Model | max_tokens | disabled TTFT | worst-case TTFT | ratio |
| :--- | :--- | :--- | :--- | :--- |
| 4B | 256 | 46 ms | 893 ms | 19.4× |
| 4B | 2048 | 24 ms | 144 ms | 5.95× |
| 27B | 256 | 181 ms | 1453 ms | 8.0× |
| 27B | 2048 | 86 ms | 427 ms | 5.0× |

  Root cause: each distinct config registers its vectors into the GPU table on the
  request's first forward pass. E2EL shows this is localized to TTFT rather than
  persistent.
  
- 27B worst-case E2EL regression is +22% at max_tokens=2048 with all 16 requests carrying distinct steering configs. Driven by per-request registration in TTFT. Mitigation path in the Optimization roadmap below.

- CUDA graphs stay captured

  Single-batch latency, per step, graphs vs eager, both with steering enabled:

| Model | graphs + steering | eager + steering | graph speedup |
| :--- | :--- | :--- | :--- |
| 4B | 738 ms | 4653 ms | 6.3× |
| 27B | 3246 ms | 8850 ms | 2.7× |

- VRAM cost is negligible

  Steering tables are [N_slots × hidden_dim] per hook point per layer. On 4B with
  max_steering_configs=32, measured at ~0.5 MB per configured slot → ~16 MB total,
  <0.02% of an H100 NVL's 94 GB. The 27B numbers scale with hidden_dim but remain
  under 0.1% of VRAM at the same slot count.

### Optimization roadmap/expected TTFT mitigations
- Minimize synchronous host-to-device transfers in register_config's new-row path
- Move name→vector resolution to workers so named vectors skip registration
- Custom Triton kernel for indexed scatter into steering tables (batches kernel launches)

## Staging plan
1. Core per-request steering on Gemma 3 as a base model (chosen for Gemma Scope existence):
  - Custom op
  - Buffer registration
  - Config types and hashing
  - SteeringManager
  - SteeringModelRunnerMixin
  - scheduler admission
  - phase separation
  - prefix cache keying
  - Streaming continuation
  - Runtime docs
  - CLI flags of --enable-steering and --max-steering-configs
  (PR 1 will be the most review surface, but minimizes to what is needed for the core functionality)
2. Global Steering HTTP API:
  - GET /v1/steering: returns layers currently configured
  - GET /v1/steering/layers: returns layers and hook points available
  - POST /v1/steering/set: sets global steering vectors
  - POST /v1/steering/clear: clears global steering config and resets prefix cache
  - CLI flag --steering-api-key for gating global API
3. Named steering modules:
  - /v1/steering/modules/register: registers steering config at a given name
  - /v1/steering/modules/unregister: clears a named steering module
  - /v1/steering/modules/: lists registered modules
  - CLI flag --steering-modules to specify path to store/load modules from
4. Distributed execution:
  - Duplicated steering tables on tensor parallel ranks
  - Only materialize required vectors on pipeline parallel ranks
5. Model wiring:
  - Adding steering functionality to remaining decoder families besides Gemma 3

## Out of scope
- Does not currently support steering speculative decoding draft models
- Does not currently support the v2 engine

Both of these can be added in follow up PRs but are deferred until the RFC has progressed further to minimize surface area.

## Prior art

Every prior in-tree attempt at activation steering — [#5807], [#7906],
[#12870] — shipped as a `ControlVectorRequest` port of repeng's
GGUF-format vectors: one vector per request, one hook point, v0 engine,
no cache integration, no `torch.compile`/CUDA-graph story, no HTTP
surface. A full audit of every prior thread is in
[prior.md][prior-md]; it shows zero maintainer technical
objections on record. The only substantive technical issues anywhere
in the audit are the three defects #5807's author self-reported —
nondeterminism, performance regression, and weak API shape — each of
which this design explicitly addresses (deterministic row assignment,
persistent GPU buffers behind a custom op, three-tier composition +
hook-point split).

The failure mode shared across #5807/#7906/#12870 was review starvation
rather than rejection on merit: assigned code owners held "Awaiting
requested review," merge conflicts accumulated, stale-bot closed. This
RFC enters the pipeline with a working implementation already in
researcher use, full H100 benchmarks, a staged merge plan, and a
ready-to-open draft PR — aimed at giving reviewers something concrete
to engage with before the same cycle repeats.

### Feature delta vs prior work

Columns: **P**hase split · **H**ook points · **G**lobal · per-**R**equest
· scheduler **A**dmission · prefix-**C**ache keying · **T**orch.compile
safe · **S**ervable (HTTP) · named **M**odules.
Legend: ✓ present and documented · ◐ partial or with caveats · ✗ not
present · — not applicable.

| Source | P | H | G | R | A | C | T | S | M |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [#3451] (concept only) | — | — | — | — | — | — | — | — | — |
| [#5807] / [#7906] (vLLM PRs) | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| [#12870] (vLLM PR, v1 rebase) | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| EasySteer (ZJU-REAL fork) | ✗ | ◐ | ✓* | ✓* | ✓ | ◐ | ◐ | ✓ | ✗ |
| [#36998] Phase 1 (observation) | n/a | ✓ | n/a | n/a | n/a | n/a | ✓ | ✓ | ✗ |
| repeng / llama.cpp CV | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| nnsight / TransformerLens | n/a | ✓ | n/a | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Goodfire Ember | — | — | — | ✓ | — | — | — | ✓ | ✓ |
| **This RFC** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

`*` — EasySteer supports G and R but forbids combining them: when a
server-level config is active, per-request steering requests are
rejected. This RFC composes global + per-request additively.

### Relationship to EasySteer

[EasySteer][easysteer] (ZJU-REAL group; `vllm-steer` submodule at
commit `21d7a9f`, "Server-level steering with CUDA graphs") is the
closest prior work and where the in-tree PRs migrated to after
stalling. Its authors have independently landed several capabilities
this RFC shares: scheduler admission keyed on a `max_steer_vectors`
budget, an HTTP surface with both startup-time (`--steer-vector-path`)
and runtime (`POST /v1/steering`) configuration, and a
CUDA-graph-compatible execution path for server-level steering
pre-baked before capture. Those are real engineering contributions and
the overlap is credit to EasySteer's authors.

The remaining differentiators, verified against EasySteer's code:

1. Phase-split vectors (P): EasySteer's `prefill_trigger_tokens` /
    `generate_trigger_tokens` are *position gates* on a single shared
    vector — users cannot specify a distinct prefill vector and a
    distinct decode vector. This RFC's three-field design (base +
    prefill + decode, additively composed) is strictly more expressive.
2. Hook-point split (H): EasySteer's `WRAPPER_REGISTRY` ships
    `decoder_layer` (block output) and `moe_layer` (router gate); the
    attention and MLP wrappers are in the file but commented out as
    future work. This RFC ships `pre_attn` / `post_attn` / `post_mlp`
    as first-class hook sites with per-site steering tables.
3. Additive global + per-request composition (G/R): EasySteer
    rejects per-request steering when a server config is active. This
    RFC composes the three tiers (base / prefill-only / decode-only,
    each at server and request scope) additively, so an operator-set
    vector can coexist with per-request research vectors in the same
    batch.
4. Cache keying on full config (C): EasySteer hashes the
    `steer_vector_name` string into block hashes, which is sufficient
    when names are append-only but *aliases* when a name is re-bound to
    different vectors. This RFC hashes the full effective prefill /
    decode config so re-binding invalidates correctly.
5. Named-module registry (M): EasySteer's `steer_vector_name` is a
    logging label; vectors are always loaded by file path. This RFC
    ships `POST /v1/steering/modules/register` as a runtime-registrable
    name → vectors registry with in-memory caching and no per-request
    file I/O.
6. CUDA graphs under heterogeneous per-request configs (T):
    EasySteer's capture benefit is demonstrated for the global
    server-level path; the per-request path has not been characterized.
    This RFC's benchmarks capture full graphs under the worst case —
    `per_request_n16`, sixteen distinct configs in one batch — and
    show 6.3× (4B) / 2.7× (27B) graph speedup over eager (see
    Performance).

In-tree vs out-of-tree is an additional non-technical delta: EasySteer
ships as a `vllm-steer` submodule users must install in place of stock
vLLM, diverging from upstream improvements. This RFC is the path to
the feature living on `main`.

### Alternatives considered

- Python forward hooks (repeng / nnsight / TransformerLens pattern):
Not `torch.compile`-safe, serializes around Python, order-of-magnitude
slower than fused execution. This is the baseline the "1–2 orders of
magnitude researcher speedup" citation in Motivation compares
against.
- Port llama.cpp's control vectors (scope of #3451/#5807/#7906/#12870):
Single vector per request, single hook point, no phase split, no
cache integration, no compile/graph story. The self-reported defects
of #5807 are structural to that design.
- Out-of-tree vLLM fork (EasySteer pattern): Proves demand but
forces users to replace their vLLM install and diverges from upstream
improvements. This RFC is explicitly the alternative to that fork
path.
- NCCL-based steering-table sharing across TP ranks: Rejected in
favor of deterministic-row-assignment with per-rank duplicated
tables, because steering tables are small (<0.02% of VRAM at
`max_steering_configs=32`) and NCCL on the steering hot path would
add a synchronization point with no memory benefit worth the cost.

## Related links
- Author's blog post with full benchmarks and design rationale:
<https://www.rhizonymph.com/blog/activation-steering>
- [Prior art audit][prior-md]
- [EasySteer][easysteer]

[prior-md]: https://github.com/RhizoNymph/vllm/blob/feat/steering/upstream/prior.md
[easysteer]: https://github.com/ZJU-REAL/EasySteer
[#3451]: https://github.com/vllm-project/vllm/issues/3451
[#5807]: https://github.com/vllm-project/vllm/pull/5807
[#7906]: https://github.com/vllm-project/vllm/pull/7906
[#12870]: https://github.com/vllm-project/vllm/pull/12870
[#36998]: https://github.com/vllm-project/vllm/issues/36998
