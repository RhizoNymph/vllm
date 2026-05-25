# vllm-activation-reward-producer

A capture-consumer plugin that turns residual-stream activations into a
scalar reward for online RL. The reward is the cosine similarity between
a mean-pooled slice of the generated span and a pre-derived reference
direction, scaled and passed through a nonlinearity:

```
reward = nonlinearity(scale * cos(mean(activations[slice]), reference_vector))
```

The architectural value of scoring inside vLLM (instead of shipping raw
activation tensors to an external scorer) is bandwidth: a
`(generated_len, hidden_size)` bf16 tensor is tens of KB per rollout; a
scalar reward is 8 bytes. At RL-loop scale that difference matters.

The consumer returns a **diagnostic dict** as `CaptureResult.payload`,
not a bare float, so the RL trainer can monitor drift signals (§ _Vector
drift_ below) without a second capture path.

## Install

```bash
pip install -e .
```

Registers the class under the name `activation_reward` in the
`vllm.capture_consumers` entry-point group.

## Usage

### Engine-side config

Layer, hook, reference vector, scale, nonlinearity, and position slice
are pinned at engine-init time. Clients cannot alter them mid-run — the
per-request spec is empty-only.

**Python API** (recommended — nested `position_slice` is not
representable in the CLI shorthand):

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-8B",
    capture_consumers=[
        {
            "name": "activation_reward",
            "params": {
                "layer": 12,
                "hook": "post_mlp",
                "vector_path": "/models/happy/sadness.pt",
                "position_slice": {"start": 10, "end": None, "stride": 1},
                "scale": 5.0,
                "nonlinearity": "tanh",
                "dtype": "float32",
            },
        },
    ],
)
```

**CLI** (only flat scalar params — defaults apply to `position_slice`):

```bash
vllm serve meta-llama/Llama-3-8B \
    --capture-consumers activation_reward:layer=12,hook=post_mlp,vector_path=/models/happy/sadness.pt,scale=5.0,nonlinearity=tanh
```

### Parameters

| Field | Type | Default | Purpose |
|---|---|---|---|
| `layer` | `int` | required | Layer index to capture at. |
| `hook` | `str` | required | One of `pre_attn`, `post_attn`, `post_mlp`, `mlp_in`, `mlp_out`. |
| `vector_path` | `str` | required | Path to a `.pt` file holding a 1-D tensor of shape `(hidden_size,)`. L2-normalized at load. |
| `position_slice` | `dict` | `{start: 10, end: null, stride: 1}` | Applied to the `all_generated` span before mean-pooling. |
| `scale` | `float` | `1.0` | Multiplicative factor on the raw cosine. |
| `nonlinearity` | `str` | `"tanh"` | `tanh`, `sigmoid`, or `identity`. |
| `dtype` | `str` | `"float32"` | Compute dtype for the cosine (`float32` or `float64`). Upcasts bf16/fp16 activations before the dot-product. |

### Per-request opt-in

Clients opt in by listing the consumer in `sampling_params.capture` with
an empty value:

```python
from vllm import SamplingParams

sampling_params = SamplingParams(
    max_tokens=128,
    capture={"activation_reward": {}},
)

[out] = llm.generate(["Tell me about your day."], sampling_params)
payload = out.capture_results["activation_reward"].payload
# {
#   "reward": 0.83,
#   "cos": 0.31,
#   "act_norm": 4.12,
#   "num_positions": 118,
#   "status": "ok",
# }
```

Requests without `activation_reward` in `capture` are not scored. A
non-empty spec (e.g. `{"layer": 99}`) is rejected with a
`CaptureValidationError` / HTTP 400 — clients cannot change the training
signal.

### Multi-feature setups

One instance per reference vector. Use `instance_name` to disambiguate:

```python
llm = LLM(
    model="meta-llama/Llama-3-8B",
    capture_consumers=[
        {
            "name": "activation_reward",
            "instance_name": "sadness_reward",
            "params": {
                "layer": 12, "hook": "post_mlp",
                "vector_path": "/models/happy/sadness.pt",
            },
        },
        {
            "name": "activation_reward",
            "instance_name": "anger_reward",
            "params": {
                "layer": 18, "hook": "post_attn",
                "vector_path": "/models/happy/anger.pt",
            },
        },
    ],
)

sampling_params = SamplingParams(
    max_tokens=128,
    capture={"sadness_reward": {}, "anger_reward": {}},
)
```

Layer and hook can differ per instance — each instance captures its own
`(layer, hook)`. Composite rewards (e.g. `max(sadness, anger)`) are the
RL loop's job, not the consumer's.

## RL-loop integration

The consumer produces rewards; the RL trainer consumes them and updates
weights externally. vLLM supports hot-loading updated weights via the
primitives documented in [docs/features/sleep_mode.md](../../../docs/features/sleep_mode.md).
A single iteration looks like:

```python
# 1. Rollout — rewards come back on RequestOutput.capture_results
sp = SamplingParams(max_tokens=128, capture={"activation_reward": {}})
outs = llm.generate(prompts, sp)
rewards = [o.capture_results["activation_reward"].payload["reward"] for o in outs]

# 2. External RL update (TRL / verl / custom). Not part of this plugin.
trainer.step(prompts, [o.outputs[0].text for o in outs], rewards)

# 3. Push updated weights back to vLLM
llm.sleep(level=2)                              # free weights + KV
llm.wake_up(tags=["weights"])                   # realloc weight buffers
llm.collective_rpc("reload_weights")            # trainer -> vLLM
llm.wake_up(tags=["kv_cache"])                  # realloc KV
```

Enable `enable_sleep_mode=True` on the `LLM(...)` constructor to permit
the sleep/wake cycle.

## Vector drift — the failure modes this plugin cannot fix

Cosine-to-a-fixed-vector reward has two known failure modes once the
policy starts training against the signal.

### 1. Geometry shift

Weight updates rotate and rescale the residual stream. The reference
direction that corresponded to a feature at iteration 0 becomes a less
meaningful axis as training proceeds. The reward signal decorrelates
from actual behavior.

**Symptom**: cosine distributions narrow over training steps and stop
responding to obvious in-distribution / out-of-distribution inputs.

### 2. Reward hacking via the direction itself

Cosine-as-reward creates a direct gradient incentive to produce
high-cosine activations. The policy learns to *occupy the reward axis*
rather than *produce the underlying behavior* the reference vector was
a proxy for.

**Symptom**: reward climbs monotonically while rollouts look degenerate
to a human reader.

## Detecting drift

Use the diagnostic payload the consumer returns:

- **`cos` distribution.** Track mean and variance over training steps.
  Monotone rise with narrowing variance strongly suggests reward
  hacking.
- **`act_norm`.** Track the L2 norm of the mean activation. Drift
  indicates geometry shift.
- **Canonical holdout.** Freeze a labeled rollout set (known-sad,
  known-neutral, known-happy). Score it every N training steps. If the
  inter-class gap collapses, the reference direction is dying.

## Frozen-scorer deployment (recommended for production)

Both failure modes come from the *policy's activation space changing
under training*. The fix is to decouple scoring from the policy:

- **Policy engine** — the vLLM instance that takes weight updates every
  RL iteration. Serves `generate` for rollouts. **Does not run
  `activation_reward`.**
- **Scorer engine** — a separate vLLM instance pinned at a frozen weight
  snapshot (no `reload_weights` ever). Runs `activation_reward`.
  Rollouts produced on the policy engine are re-run through the scorer
  engine purely to read activations; the reward payload off the
  scorer's `capture_results` is what the RL trainer consumes.

Two engines cost 2× VRAM but eliminate both failure modes by
construction — a frozen scorer's activation space does not change.
No code change in this plugin; which engine it is loaded into is the
only thing that differs.

For smaller experiments, single-engine mode is fine — just watch the
canonical holdout set and be prepared to stop training or re-derive the
reference direction when the inter-class gap collapses.

## Reference-vector derivation matters

This plugin loads a tensor; it does not care how you derived it. But
different derivations have very different drift resistance:

- **Mean of activations from labeled positive examples** — simplest,
  most drift-prone. Easy to start with; expect to replace it.
- **Linear probe direction** (trained to classify positive vs negative
  at this `(layer, hook)`) — trained against a discriminative task,
  more robust.
- **SAE feature direction** (from a pretrained sparse autoencoder at
  this `(layer, hook)`) — most interpretable, requires the SAE.

The plugin L2-normalizes on load, so pre-normalization of the saved
tensor is not required.

## Known constraints

- **TP=1, PP=1 only.** Same limitation the built-in filesystem consumer
  has. Multi-rank collection is out of scope for the current capture
  framework (design doc invariant 8).
- **One instance per reference vector.** Multi-feature = multiple
  instances with distinct `instance_name`s.
- **Payload is a dict, not a bare float.** Downstream code should read
  `result.payload["reward"]`, not `float(result.payload)`.
- **Direct `CaptureSink`, not `CaptureConsumer`.** The framework's
  `_BatchedAdapter` hard-codes `payload=None` on success; to emit a
  diagnostic dict we bypass it and implement the sink protocol
  directly. See `activation_reward_producer/__init__.py` for the
  buffer-and-compute pattern (≈50 lines, mirrors `_BatchedAdapter`
  minus the payload constraint).

## Testing

Run `python test.py` in this directory for a standalone lifecycle
smoke test (uses a `MagicMock` `VllmConfig`; no engine required).
