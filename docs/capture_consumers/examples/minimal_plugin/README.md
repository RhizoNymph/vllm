# vllm-sum-consumer

Minimal example capture consumer plugin for vLLM. Records the sum of
every captured tensor.

## Install

```bash
pip install -e .
```

## Usage

```bash
vllm serve my-model --capture-consumers sum:layers=[0,15,31]
```

The consumer registers under the name `sum` in the
`vllm.capture_consumers` entry-point group. See
`docs/capture_consumers/plugin_authoring.md` for a full tutorial on
writing capture consumer plugins.
