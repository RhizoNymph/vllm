# Prior Work

Exhaustive survey of prior work related to activation steering in vLLM.
Organized as:

1. [In-tree vLLM threads](#1-in-tree-vllm-threads) — all RFCs, issues, and PRs
   in `vllm-project/vllm` that touch steering, control vectors, or
   activation intervention.
2. [Out-of-tree implementations](#2-out-of-tree-implementations) — libraries,
   forks, and frameworks that implement steering outside vLLM.
3. [Academic prior art](#3-academic-prior-art) — the research methods this
   runtime is designed to serve.
4. [Delta summary](#4-delta-summary) — a single table enumerating what each
   prior work provides and what the proposed runtime adds.

This document is the long-form reference for the `Prior Art` and
`Alternatives Considered` sections of the RFC. The RFC cites rows of the
[delta summary](#4-delta-summary) directly.

> **Citation convention.** Links are provided where they are stable and the
> author is confident. Items marked `<!-- url: TBD -->` are correct
> bibliographically but need a URL filled in by the submitter (usually
> arXiv or a project page).

---

## 1. In-tree vLLM threads

### 1.1 [#3451] — Feature request: Control vectors

- **Opened:** Mar 17 2024 by @generalsvr.
- **Closed:** Nov 2024 as "not planned". Labels: `stale`, `feature request`.
- **Proposal:** Port llama.cpp's control-vector concept (see §2.2) into
  vLLM as a first-class feature. No concrete design attached; the request
  references repeng (§2.1) and llama.cpp's implementation as precedent.
- **Why it closed — verbatim evidence:** No closing comment from any
  maintainer was recoverable, and no assignee was ever attached. The only
  machine-visible signals are the `stale` label and the "Closed as not
  planned" state flag.
- **Why it closed — inference:** The combination of `stale` label, no
  assignee, no maintainer comment, and two concurrently-live
  implementation PRs (#5807, #7906) points to a **stale-bot auto-close of
  an orphaned feature request**, not an explicit maintainer veto of the
  idea. The "not planned" label in this project is routinely applied by
  stale automation and should not be read as a scope decision on the
  merits. **There is no verbatim record of any maintainer saying the
  feature should not exist.**
- **How the current proposal differs:** The current proposal is a
  concrete design for every engine-level concern that a pre-v1 request
  could not have addressed — phase-aware admission, prefix-cache key
  separation, `torch.compile`/CUDA-graph safety, three-tier composition.
  Where #3451 was a concept without a design, this RFC is a design with
  runtime correctness arguments for each hot-path concern.

[#3451]: https://github.com/vllm-project/vllm/issues/3451

### 1.2 [#5807] — Draft: Control Vector Support (first attempt)

- **Author:** @raywanb.
- **Opened:** Jun 2024 as draft.
- **Closed:** Aug 2024, author self-closed.
- **Proposal:** Per-request `ControlVectorRequest` wired into the v0
  engine. Vectors loaded from GGUF (repeng format). One control vector
  per request; `enable_control_vector`, `normalize_control_vector`,
  `max_control_vectors` CLI flags.
- **Why it closed — verbatim self-diagnosis from the author:**
  > "Adding Control Vector support as seen from [repeng] But currently
  > it is still buggy."

  Three self-listed defects in the PR description:

  1. "need help looking over how control vector requests (similar to
     lora requests) are passed since the same two inputs yielded two
     different outputs" — **nondeterminism**.
  2. "seems to impact inference speed" — **performance regression**.
  3. "could use a better structure (esp when ControlVectorConfig class
     is not really used)" — **weak API shape**.

  Review was requested from @DarkLight1337 and @22quinn; no substantive
  review landed. The author self-closed after asking for plumbing help
  (parity with the LoRA request path) that never arrived.
- **Why it closed — inference:** Author self-abandonment of an
  admittedly-buggy draft after maintainer review silence. Not a
  maintainer rejection of the feature.
- **How the current proposal addresses each self-reported defect:**
  - Nondeterminism: `SteeringManager`'s `config_to_row` allocation is
    deterministic across replicas (the distributed roadmap explicitly
    makes this a contract). Per-request steering is applied via an
    indexed gather with fixed buffer shapes — no source of
    nondeterminism remains.
  - Performance regression: the custom-op design with persistent GPU
    buffers updated between steps (not per-forward hooks) is what
    keeps the steady-state decode overhead at the single-digit-percent
    level cited in the RFC. Benchmarks are in the blog post and will
    be in PR 1.
  - Weak API shape: the three-tier composition, three hook points,
    named modules, and HTTP surface are all downstream of deliberate
    design iteration; the API is not a ported LoRA analogue.
- **Limitations vs current proposal (scope):**
  - Single vector per request (no three-tier composition).
  - Applied at a single hook point (no `pre_attn`/`post_attn`/`post_mlp`
    split).
  - No global (server-wide) steering.
  - No HTTP API for runtime control.
  - v0 engine only. No phase awareness.
  - No prefix-cache key integration; APC interaction was not analyzed.
  - No `torch.compile` or CUDA-graph story.
  - No scheduler admission / capacity accounting.

[#5807]: https://github.com/vllm-project/vllm/pull/5807

### 1.3 [#7906] — Control Vector Support (retry)

- **Author:** @raywanb.
- **Opened:** Aug 2024.
- **Closed:** Mar 27 2025 by stale-bot.
- **Proposal:** Refreshed version of #5807 rebased on more recent main.
  Same API surface: `ControlVectorRequest`, GGUF loading, per-request
  only, one vector per request.
- **Why it closed — verbatim:** Stale-bot sequence:

  > "This pull request has been automatically marked as stale because
  > it has not had any activity within 90 days."

  followed by:

  > "This pull request has been automatically closed due to inactivity.
  > Please feel free to reopen if you intend to continue working on it."

  Review was requested on Nov 26 2024 from five code owners —
  @zhuohan123, @youkaichao, @alexm-redhat, @comaniac, @njhill — and
  every one of them remained "Awaiting requested review" until close.
  @mergify posted the standard merge-conflict notice. No substantive
  technical feedback was ever left on the PR.
- **Why it closed — inference:** **Zero reviews landed from five
  assigned code owners over four months.** The closure was automated
  staleness driven by maintainer-capacity exhaustion and unresolved
  merge conflicts, not any maintainer objection to the feature. This is
  the single strongest piece of evidence for the `Maintenance ownership`
  risk called out in the RFC.
- **Limitations vs current proposal (scope):** identical to #5807 —
  per-request only, no phase split, no cache integration, no
  compile/graph story, no HTTP API, no scheduler integration.

[#7906]: https://github.com/vllm-project/vllm/pull/7906

### 1.4 [#12870] — Control Vector Support (community refresh)

- **Author:** @yuu-biz.
- **Opened:** Feb 2025.
- **Closed:** Sep 11 2025 by stale-bot.
- **Proposal:** Community refresh of #7906 with merge conflicts
  resolved against v1 engine. Same `ControlVectorRequest` shape.
- **Why it closed — verbatim:** Stale-bot sequence identical to #7906
  (marked stale Aug 11 2025, closed Sep 11 2025). The same five code
  owners (@zhuohan123, @youkaichao, @alexm-redhat, @comaniac, @njhill)
  held "Awaiting requested review." @mergify flagged `needs-rebase` on
  May 12 2025.

  Author rationale before stalling, verbatim (Mar 10 2025):

  > "I haven't been able to work on this PR due to other matters, but
  > I'll check the Slack now for PR review."

- **Why it closed — inference:** **Identical failure mode to #7906** —
  review starvation from assigned code owners, compounded by author
  bandwidth loss and merge-conflict rot. No maintainer technical
  objection is on record.
- **Downstream outcome:** After this PR stalled, the "EasySteer" out-of-
  tree fork (ZJU-REAL group; see §2.3) took the patches and continued
  development outside `vllm-project/vllm` — signal that community demand
  outlasted the review pipeline.
- **Limitations vs current proposal:** same narrow scope as #7906.

[#12870]: https://github.com/vllm-project/vllm/pull/12870

### 1.5 [#38881] — (accidental upstream PR)

- **Author:** @RhizoNymph (this RFC's author).
- **Opened and closed:** Apr 3 2026 (same day).
- **Why it closed — verbatim, from the PR description:**

  > "PR made erroneously before I'm ready, will make sure this cant
  > happen again."

  And from the closing comment:

  > "I'm so sorry I did not mean to open this yet this was supposed to
  > target my branch, I will make sure this cant happen again."

- **Why it closed — inference:** Pure operator error — a topic branch
  was pushed against the wrong base. **Not a substantive PR and not a
  maintainer rejection.** The only review activity was an automated
  @gemini-code-assist pass that flagged four unrelated attribute-name
  bugs in `gpu_model_runner.py` (`rs.sampling_params` vs `rs.params`
  on `CachedRequestState`) — those are not features of the design.
- **Status vs current proposal:** Not superseded — simply never the
  intended upstream artifact. Included here only for transparency about
  the author's GitHub footprint on `vllm-project/vllm`.

[#38881]: https://github.com/vllm-project/vllm/pull/38881

### 1.6 [#36998] — RFC: Observation Plugin for Intercepting & Routing on Activations

- **Author:** DDDDarrenWB; the author of this RFC is a co-author.
- **Opened:** Mar 2026.
- **State:** Open, active.
- **Proposal:** A plugin system for intercepting activations via
  `register_forward_hook`-style primitives. Phase 1 is read-only
  prefill-only observation with ABORT/CONTINUE control flow; Phase 2
  explicitly defers activation-stream *mutation*.
- **Relationship to this RFC:** Sibling, not duplicate. The runtime
  proposed here is the substrate that Phase 2 of #36998 needs (because
  `register_forward_hook` is not `torch.compile`-safe and cannot key the
  prefix cache). This RFC additionally exposes a direct non-dynamic API
  that #36998 explicitly does not cover — the research-and-operator
  workflow described in the RFC's Motivation section.
- **Coordination status:** Co-authorship confirmed with the #36998
  author; this RFC is agreed to be the venue for the runtime and the
  direct API, with #36998 Phase 2 consuming the runtime for
  dynamic-mutation plugin support.

[#36998]: https://github.com/vllm-project/vllm/issues/36998

### 1.7 Other in-tree threads considered and ruled out

The following were examined during the duplicate-work check and confirmed
unrelated despite surface-level term overlap:

- `residual stream` hits: `#38978`, `#38479`, `#36101`, `#34047`, `#34046`,
  `#33123`. All are KV-cache, RMSNorm, MTP, or APC work — no steering
  semantics.

No other vllm-project/vllm threads propose or implement activation
steering or a related mutation API.

---

## 2. Out-of-tree implementations

### 2.1 repeng (vgel/repeng)

- **Author:** Theia Vogel.
- **Approach:** Python library. Trains control vectors from contrastive
  prompt pairs, applies them via HuggingFace `register_forward_hook` on
  the residual stream of each decoder layer. Exports to a GGUF format
  consumed by llama.cpp and prior vLLM attempts.
- **Strengths:** Mature trainer; clear vector-export format that has
  become a de facto standard; broad adoption in the alignment research
  community.
- **Limitations vs in-tree runtime:**
  - Inference path is HuggingFace forward hooks — not `torch.compile`-
    safe, serializes around Python state, order-of-magnitude slower
    than vLLM's fused execution.
  - Single hook point per layer (post-decoder residual); does not
    distinguish `pre_attn` / `post_attn` / `post_mlp`.
  - No server / API surface — it is a library.
  - No prefix caching; every prompt runs full prefill.
  - No batching of heterogeneous steering configs.
- **Relationship:** This is the baseline the RFC's Motivation section
  compares against when it cites "1–2 orders of magnitude" researcher
  speedup. The GGUF export format should be supported as an input format
  for the `--steering-modules` loader in a follow-up PR.
- **URL:** https://github.com/vgel/repeng

### 2.2 llama.cpp control vectors

- **PR:** llama.cpp `#5970` <!-- url: verify canonical PR number before citing -->
- **Approach:** C++ implementation of repeng-format control vectors
  applied during inference. One vector per request; no phase split.
- **Limitations vs in-tree runtime:**
  - Engine-specific to llama.cpp; not applicable to server deployments
    built on vLLM.
  - Same single-hook, single-vector, no-phase-split limits as repeng.
  - Referenced by [#3451] as the precedent maintainers were asked to
    match.
- **Relationship:** Establishes that the concept has shipped in a
  comparable engine. The proposed runtime is a superset.

### 2.3 EasySteer (ZJU-REAL fork)

- **Authors:** ZJU-REAL group (Zhejiang University).
- **Approach:** Out-of-tree vLLM fork that continues the work of
  [#12870] after it went stale. Ships as an `EasySteer` top-level repo
  with a `vllm-steer` submodule (its own repo at
  `ZJU-REAL/EasySteer-vllm-v1`) containing the forked engine code,
  plus a Python SDK (`easysteer/`) layered on top for training and
  analysis tooling around per-request control vectors.
- **Fork basis verified at:** `vllm-steer` submodule commit
  `21d7a9f61d879be6b66347ba122fd7fd815500be` (2026-03-31), titled
  "Server-level steering with CUDA graphs (2.3x speedup)." This is
  materially more advanced than #12870; the characterization below
  reflects a direct code audit of that commit, not the README.
- **What EasySteer has (verified in code):**
  - *Scheduler admission keyed on `max_steer_vectors`*, LoRA-style.
    `vllm-steer/vllm/v1/core/sched/scheduler.py:606-619` refuses to
    admit a new request whose steering config would exceed capacity.
  - *Startup-time and runtime server-level steering.*
    `--steer-vector-path` CLI flag loads a server default; `POST
    /v1/steering` mutates scale at runtime; `GET /v1/steering` reports
    current state. `vllm-steer/vllm/entrypoints/openai/api_server.py`.
  - *Per-request steering via `extra_body.steer_vector_request`.*
    `vllm-steer/vllm/steer_vectors/request.py:159-176`.
  - *Prefix-cache integration via block-hash extra keys.*
    `_gen_steer_vector_extra_hash_keys()` in
    `vllm-steer/vllm/v1/core/kv_cache_utils.py:470-482` adds the
    `steer_vector_name` string into block-hash computation. **Caveat:**
    only the name is hashed, not the vector contents, so re-binding a
    name to different vectors aliases across the cache.
  - *CUDA graph capture for the server-level path.* Server-level
    steering is pre-baked before capture with `prefill_trigger_tokens
    = [-1]` / `generate_trigger_tokens = [-1]` to give the graph a
    fixed intervention shape (`vllm-steer/vllm/v1/worker/
    steer_vector_model_runner_mixin.py:52-94`). This is the basis of
    the "2.3× speedup" claim.
  - *OpenAI-compatible HTTP surface* for both the steering endpoints
    and the chat/completion path.
- **What EasySteer does not have (verified in code):**
  - *Phase-split vectors.* `prefill_trigger_tokens` and
    `generate_trigger_tokens` are per-token position gates on a single
    shared vector per `VectorConfig`, not distinct prefill/decode
    vector fields (`vllm-steer/vllm/steer_vectors/worker_manager.py:
    127-155`). Users cannot specify one vector for prefill and a
    different vector for decode.
  - *Hook-point split.* `WRAPPER_REGISTRY` in
    `vllm-steer/vllm/steer_vectors/config.py:212-254` enables only
    `decoder_layer` (block output) and `moe_layer` (router gate); the
    attention and MLP wrappers are present in the file but commented
    out as future work. There is no `pre_attn` / `post_attn` /
    `post_mlp` distinction at intervention time.
  - *Additive global + per-request composition.* `vllm-steer/vllm/
    entrypoints/openai/engine/serving.py:780` rejects per-request
    steering when a server-level config is active with the message
    "Per-request steer_vector_request is not allowed when server-level
    steering is active." The two tiers are mutually exclusive, not
    additive.
  - *Runtime-registrable named modules.* `steer_vector_name` is a
    logging label only (`vllm-steer/vllm/steer_vectors/request.py`);
    vectors are always resolved by file path
    (`steer_vector_local_path`). There is no register/unregister
    endpoint and no in-memory name→vectors table.
  - *CUDA graph capture characterized under heterogeneous per-request
    configs.* The pre-bake-before-capture trick in
    `steer_vector_model_runner_mixin.py` is specifically for the
    global path; the per-request path uses per-forward-pass position-
    tensor recomputation (`vllm-steer/vllm/steer_vectors/algorithms/
    template.py:191-234`) and the interaction with full-graph capture
    under distinct per-request configs in one batch is not
    demonstrated.
- **Cross-reference to RFC delta:** Maps to the EasySteer row
  `✗ ◐ ✓* ✓* ✓ ◐ ◐ ✓ ✗` in §4 below. The `*` marks the
  G/R mutual-exclusivity gotcha — both columns show ✓ individually,
  but they cannot combine in one runtime state.
- **Relationship:** Demonstrates sustained community demand for
  in-tree steering and has independently solved parts of the problem
  space (admission, global HTTP API, CUDA graphs for the global
  path). An in-tree runtime subsumes EasySteer's capabilities, adds
  the five differentiators above, and removes the fork-replacement
  tax. The GGUF export format should be supported as an input format
  for the `--steering-modules` loader in a follow-up PR.
- **URL:** https://github.com/ZJU-REAL/EasySteer
  (submodule: https://github.com/ZJU-REAL/EasySteer-vllm-v1)

### 2.4 nnsight

- **Author:** NDIF (National Deep Inference Facility) / Bau lab.
- **Approach:** PyTorch-level intervention framework. Executes a
  declarative intervention graph against a model forward pass; supports
  reading and writing at arbitrary module points.
- **Strengths:** Expressive; standard tool in interpretability research.
- **Limitations vs in-tree runtime:** Not an inference server. No
  batching of heterogeneous configs. No prefix caching. Not designed
  for production serving.
- **Relationship:** Complementary — researchers use nnsight to *derive*
  steering vectors and the proposed runtime to *deploy* them at scale.
- **URL:** https://github.com/ndif-team/nnsight

### 2.5 TransformerLens

- **Author:** Neel Nanda et al.
- **Approach:** Mechanistic-interpretability library with a hook-point
  API that inspired the `pre_attn` / `post_attn` / `post_mlp` naming
  used here.
- **Relationship:** Same as nnsight — research-time tool, not a serving
  runtime. The hook-point vocabulary is adopted because it matches what
  researchers already think in.
- **URL:** https://github.com/TransformerLensOrg/TransformerLens

### 2.6 Goodfire Ember

- **Authors:** Goodfire AI.
- **Approach:** Commercial API for SAE-feature-level steering of hosted
  models. Closed-source.
- **Limitations vs in-tree runtime:** Vendor-hosted only; cannot be run
  on private deployments or custom models; no local-model story.
- **Relationship:** Establishes commercial demand for the workflow the
  runtime enables. Not a substitute for an in-tree implementation.
- **URL:** https://goodfire.ai

### 2.7 Obvs / other intervention tooling

Various smaller libraries (`inseq`, `pyvene`, `baukit`) offer
intervention primitives at the PyTorch level. All share the same
limitations relative to an inference engine: no batching across configs,
no cache integration, not `torch.compile`-safe.

---

## 3. Academic prior art

Methods this runtime is designed to serve. Included so reviewers who are
not familiar with the research landscape can see what class of work the
feature unblocks at production scale.

### 3.1 Activation Addition (ActAdd)

- **Turner et al., 2023.** "Steering GPT-2-XL by adding an activation
  vector."
- Defines the basic ActAdd primitive: compute the difference in residual
  activations between a "positive" and "negative" prompt pair, add
  scaled difference to target-prompt activations at inference time.
- **URL:** <!-- url: TBD — arXiv -->

### 3.2 Representation Engineering (RepE)

- **Zou et al., 2023.** "Representation Engineering: A Top-Down Approach
  to AI Transparency."
- Generalizes ActAdd to reading and control of high-level
  representations (honesty, sycophancy, etc.). Introduces
  `RepReadingPipeline` / `RepControlPipeline`.
- **URL:** <!-- url: TBD — arXiv -->

### 3.3 Contrastive Activation Addition (CAA)

- **Panickssery et al., 2023.** "Steering Llama 2 via Contrastive
  Activation Addition."
- Averages activation differences across many contrastive pairs; one of
  the most cited recipes for producing control vectors for open-weight
  models.
- **URL:** <!-- url: TBD — arXiv -->

### 3.4 Inference-Time Intervention (ITI)

- **Li et al., 2023.** "Inference-Time Intervention: Eliciting Truthful
  Answers from a Language Model."
- Applies linear interventions to specific attention heads at inference.
- **URL:** <!-- url: TBD — arXiv -->

### 3.5 SAE-feature steering

- **Templeton et al., Anthropic, 2024.** "Scaling Monosemanticity."
  Steering by clamping sparse autoencoder feature activations.
- **Lieberum et al., DeepMind, 2024.** "Gemma Scope."  Open-weights SAEs
  trained on Gemma 2, directly usable as steering-vector sources for the
  Gemma-family architectures this runtime supports.
- **URLs:** <!-- url: TBD — transformer-circuits.pub and arXiv -->

### 3.6 What the runtime does for these methods

None of the cited methods require a novel inference primitive — they
require activation addition at specific hook points. What they lack in
practice is a batched, cache-aware, compile-safe way to deploy the
vectors they produce. That deployment story is precisely what an
in-tree vLLM runtime provides. This is the research-workflow angle
referenced in the RFC's Motivation.

---

## 4. Delta summary

Columns:
- **P**hase split (base / prefill / decode)
- **H**ook points distinguished (`pre_attn` / `post_attn` / `post_mlp`)
- **G**lobal server-wide vectors
- **R**equest-scoped vectors
- **A**dmission (scheduler capacity accounting)
- **C**ache (prefix-cache key integration)
- **T**orch.compile / CUDA graph safe
- **S**ervable (HTTP API + OpenAI protocol)
- **M**odules (named, reusable)

Legend: ✓ present and documented · ◐ partial or with caveats ·
✗ not present · — not applicable.

| Source | P | H | G | R | A | C | T | S | M |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| #3451 (concept only) | — | — | — | — | — | — | — | — | — |
| #5807 / #7906 (vLLM PRs) | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| #12870 (vLLM PR, v1 rebase) | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| EasySteer (ZJU-REAL fork) | ✗ | ◐ | ✓* | ✓* | ✓ | ◐ | ◐ | ✓ | ✗ |
| #38881 (own prior) | n/a — accidental, closed same day | | | | | | | |
| #36998 Phase 1 (observation) | n/a | ✓ | n/a | n/a | n/a | n/a | ✓ | ✓ | ✗ |
| repeng / llama.cpp CV | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| nnsight / TransformerLens | n/a | ✓ | n/a | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Goodfire Ember | — | — | — | ✓ | — | — | — | ✓ | ✓ |
| **This RFC** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

`*` — EasySteer supports G and R but forbids combining them: when a
server-level config is active, per-request steering requests are
rejected. This RFC composes global + per-request additively.

Footnotes on EasySteer's ◐ entries (full code-level detail in §2.3):
- **H:** only `decoder_layer` block output and `moe_layer` router gate
  are enabled; the attention and MLP wrappers are commented out as
  future work. No `pre_attn` / `post_attn` / `post_mlp` distinction.
- **C:** the `steer_vector_name` string is included in block-hash
  extra keys, but only the name is hashed — not the vector contents.
  Re-binding a name to different vectors aliases across the cache.
- **T:** CUDA graph capture is demonstrated for the server-level
  (global) path via pre-bake-before-capture with global triggers;
  capture behavior under heterogeneous per-request configs in one
  batch is not characterized.

The table makes the non-duplication argument mechanical: no prior work
covers the full set. The closest existing in-tree work (#7906 /
#12870) covers one column. EasySteer is the closest out-of-tree
serving work and has independently landed several columns
(A, S, and partial credit on G/R/C/T/H) but leaves five differentiators
intact (full P, full H, additive G+R, full-hash C, runtime M, and
per-request CUDA graph capture). The closest commercial alternative
(Goodfire Ember) is closed-source and vendor-hosted.

---

## 5. What this implies for upstreaming

The key finding from auditing every prior thread: **no maintainer is on
record rejecting the feature on scope, correctness, or strategy grounds.**
The only substantive technical objections anywhere in the audit are the
three self-reported defects in #5807 — nondeterminism, performance
regression, and weak API shape — and the current design explicitly
addresses each (see §1.2).

Four concrete conclusions feed back into the RFC:

1. **The "not planned" label on [#3451] does not bind this RFC.** It is
   consistent with a stale-bot auto-close of an orphaned feature request,
   not an explicit maintainer veto. No maintainer comment on that issue
   was recovered that says the feature should not exist.
2. **Every closed implementation PR stalled on the same failure mode:**
   assigned code owners never reviewed, author bandwidth eroded,
   merge-conflict rot set in, stale-bot closed. Across #5807, #7906, and
   #12870, the total count of substantive maintainer review comments
   recovered is **zero**. This is a maintenance-capacity problem, not a
   feature-quality problem.
3. **The sustained stream of narrower attempts (#5807, #7906, #12870) and
   the EasySteer fork are positive signals.** They establish durable
   community demand over 18+ months and two academic groups. EasySteer
   in particular has materially advanced past the #12870 scope (see
   §2.3) — adding server-level steering, scheduler admission, and
   CUDA graph capture for the global path — which both validates the
   engineering problem as tractable and sharpens the differentiator
   story for this RFC.
4. **Coordination with [#36998] removes the ambiguity of two parallel
   RFCs covering overlapping runtime concerns.** #36998's author is a
   fellow contributor (not a vLLM maintainer); co-authorship on the
   runtime is confirmed and documented so that reviewers see one
   runtime proposal and one dynamic-steering consumer, not two
   competing designs. This does not by itself solve the review-
   capacity problem — the mitigations for that are (a) a working
   implementation already in researcher use, (b) full H100 benchmarks,
   (c) a staged merge plan, and (d) engagement via the developer
   Slack once the draft PR opens.

The takeaway for the RFC framing is that the "why now" story is not
*"we've overcome prior rejections"* — it is *"prior attempts stalled
before they could be evaluated on merit; this one enters the review
pipeline with a committed maintainer, a design that addresses the only
on-record technical objections, and external adoption as evidence of
readiness."*
