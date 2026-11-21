# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small decoder layers written in the style of Hugging Face modeling files.

Each mirrors the residual structure of a real architecture family, using tiny
deterministic sublayers so the tests can compute every hook-point tensor by
hand. `StubAttention` stands in for an attention module (the tests' attention
predicate recognises it, as the backend recognises a vLLM `AttentionLayerBase`).

`reference(x, d_pre, d_mlp_in)` computes each hook-point tensor directly, with
`d_pre` added to the skip of the attention residual add (what a `pre_attn`
steer does) and `d_mlp_in` added to the MLP input (what an `mlp_in` steer
does).
"""

import torch
from torch import nn


class StubAttention(nn.Module):
    """Returns `(tanh(proj(x)), None)` like an HF attention module."""

    def __init__(self, hidden: int):
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return torch.tanh(self.o_proj(hidden_states)), None


class StubMLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden, 2 * hidden, bias=False)
        self.down_proj = nn.Linear(2 * hidden, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(x)))


class StubNorm(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 1.5, hidden))

    def forward(self, hidden_states):
        return hidden_states * self.weight / (1.0 + hidden_states.abs().mean(-1, True))


def is_stub_attention(module: nn.Module) -> bool:
    return isinstance(module, StubAttention)


class _Base(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.self_attn = StubAttention(hidden)
        self.mlp = StubMLP(hidden)
        self.input_layernorm = StubNorm(hidden)
        self.post_attention_layernorm = StubNorm(hidden)


class LlamaLikeLayer(_Base):
    """Pre-norm sequential block (Llama, Qwen2, Gemma, OLMoE, ...)."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def reference(self, x, d_pre=0.0, d_mlp_in=0.0):
        attn, _ = self.self_attn(self.input_layernorm(x))
        post_attn = x + d_pre + attn
        mlp_in = self.post_attention_layernorm(post_attn) + d_mlp_in
        mlp_out = self.mlp(mlp_in)
        return {
            "pre_attn": x + d_pre,
            "post_attn": post_attn,
            "mlp_in": mlp_in,
            "mlp_out": mlp_out,
            "post_block": post_attn + mlp_out,
        }


class PostNormLayer(_Base):
    """Norms on the sublayer outputs (OLMo 2/3, FlexOlmo)."""

    def __init__(self, hidden: int = 8):
        super().__init__(hidden)
        del self.input_layernorm
        self.post_feedforward_layernorm = StubNorm(hidden)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, **kwargs
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def reference(self, x, d_pre=0.0, d_mlp_in=0.0):
        attn, _ = self.self_attn(x)
        post_attn = x + d_pre + self.post_attention_layernorm(attn)
        mlp_in = post_attn + d_mlp_in
        mlp_out = self.post_feedforward_layernorm(self.mlp(mlp_in))
        return {
            "pre_attn": x + d_pre,
            "post_attn": post_attn,
            "mlp_in": mlp_in,
            "mlp_out": mlp_out,
            "post_block": post_attn + mlp_out,
        }


class ScaledDropoutLayer(_Base):
    """Granite-style residual multiplier plus Phi-3-style residual dropouts."""

    def __init__(self, hidden: int = 8):
        super().__init__(hidden)
        self.residual_multiplier = 0.25
        self.resid_attn_dropout = nn.Dropout(0.0)
        self.resid_mlp_dropout = nn.Dropout(0.0)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, **kwargs
        )
        hidden_states = residual + self.resid_attn_dropout(
            hidden_states * self.residual_multiplier
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states) * (
            self.residual_multiplier
        )
        return hidden_states

    def reference(self, x, d_pre=0.0, d_mlp_in=0.0):
        attn, _ = self.self_attn(self.input_layernorm(x))
        post_attn = x + d_pre + attn * self.residual_multiplier
        mlp_in = self.post_attention_layernorm(post_attn) + d_mlp_in
        mlp_out = self.mlp(mlp_in) * self.residual_multiplier
        return {
            "pre_attn": x + d_pre,
            "post_attn": post_attn,
            "mlp_in": mlp_in,
            "mlp_out": mlp_out,
            "post_block": post_attn + mlp_out,
        }


class InlineMlpLayer(_Base):
    """The MLP call nested inside the residual add, returned as a 1-tuple."""

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        attn_output, _ = self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return (hidden_states,)

    def reference(self, x, d_pre=0.0, d_mlp_in=0.0):
        return LlamaLikeLayer.reference(self, x, d_pre, d_mlp_in)


class ParallelLayer(_Base):
    """Phi: attention and MLP read the same normed input; one residual add."""

    def __init__(self, hidden: int = 8):
        super().__init__(hidden)
        del self.post_attention_layernorm
        self.resid_dropout = nn.Dropout(0.0)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, **kwargs
        )
        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        hidden_states = attn_outputs + feed_forward_hidden_states + residual

        return hidden_states

    def reference(self, x, d_pre=0.0, d_mlp_in=0.0):
        normed = self.input_layernorm(x)
        attn, _ = self.self_attn(normed)
        mlp_in = normed + d_mlp_in
        mlp_out = self.mlp(mlp_in)
        post_attn = x + d_pre + attn
        return {
            "pre_attn": x + d_pre,
            "post_attn": post_attn,
            "mlp_in": mlp_in,
            "mlp_out": mlp_out,
            "post_block": post_attn + mlp_out,
        }


class NeoXLikeLayer(_Base):
    """GPT-NeoX: a config flag picks a parallel or a sequential block."""

    def __init__(self, hidden: int = 8, use_parallel_residual: bool = True):
        super().__init__(hidden)
        self.use_parallel_residual = use_parallel_residual
        self.post_attention_dropout = nn.Dropout(0.0)
        self.post_mlp_dropout = nn.Dropout(0.0)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        attn_output, _ = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            **kwargs,
        )
        attn_output = self.post_attention_dropout(attn_output)

        if self.use_parallel_residual:
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output

        return hidden_states

    def reference(self, x, d_pre=0.0, d_mlp_in=0.0):
        attn, _ = self.self_attn(self.input_layernorm(x))
        post_attn = x + d_pre + attn
        source = x if self.use_parallel_residual else post_attn
        mlp_in = self.post_attention_layernorm(source) + d_mlp_in
        mlp_out = self.mlp(mlp_in)
        return {
            "pre_attn": x + d_pre,
            "post_attn": post_attn,
            "mlp_in": mlp_in,
            "mlp_out": mlp_out,
            "post_block": post_attn + mlp_out,
        }


class OptionalCrossAttentionLayer(_Base):
    """GPTBigCode: a cross-attention block whose modules exist only when the
    config adds cross-attention. Without them that path would raise, so it is
    not a path the layer can take."""

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        residual = hidden_states
        hidden_states, _ = self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + residual
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn, _ = self.crossattention(self.ln_cross_attn(hidden_states))
            hidden_states = residual + cross_attn
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states

    def reference(self, x, d_pre=0.0, d_mlp_in=0.0):
        return LlamaLikeLayer.reference(self, x, d_pre, d_mlp_in)


SUPPORTED_LAYERS = [
    LlamaLikeLayer,
    PostNormLayer,
    ScaledDropoutLayer,
    InlineMlpLayer,
    ParallelLayer,
    NeoXLikeLayer,
    OptionalCrossAttentionLayer,
]


# --- layouts the backend must refuse ----------------------------------------


class NoAttentionLayer(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.mlp = StubMLP(hidden)
        self.mlp2 = StubMLP(hidden)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(hidden_states)
        return hidden_states + self.mlp2(hidden_states)


class CrossAttentionLayer(_Base):
    """A third residual add (cross-attention) is not a standard layout."""

    def __init__(self, hidden: int = 8):
        super().__init__(hidden)
        self.cross_attn = StubAttention(hidden)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states + self.self_attn(hidden_states)[0]
        hidden_states = hidden_states + self.cross_attn(hidden_states)[0]
        return hidden_states + self.mlp(hidden_states)


class AugAssignLayer(_Base):
    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states, _ = self.self_attn(self.input_layernorm(hidden_states))
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class InPlaceLayer(_Base):
    def forward(self, hidden_states, **kwargs):
        attn, _ = self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + attn
        mlp = self.mlp(self.post_attention_layernorm(hidden_states))
        return mlp.add_(hidden_states)


class LoopLayer(_Base):
    def forward(self, hidden_states, **kwargs):
        for _ in range(2):
            hidden_states = hidden_states + self.self_attn(hidden_states)[0]
        return hidden_states + self.mlp(hidden_states)


class SplitParallelLayer(_Base):
    """The two branches are summed before the residual add, so post_attn is
    never formed and cannot be recovered without re-associating statements."""

    def forward(self, hidden_states, **kwargs):
        normed = self.input_layernorm(hidden_states)
        attn, _ = self.self_attn(normed)
        branches = attn + self.mlp(normed)
        return hidden_states + branches


class HelperMethodLayer(_Base):
    def _sublayers(self, x):
        return self.self_attn(x)[0]

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states + self._sublayers(hidden_states)
        return hidden_states + self.mlp(hidden_states)


class TwoMlpCallsLayer(_Base):
    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states + self.self_attn(hidden_states)[0]
        return hidden_states + self.mlp(self.mlp(hidden_states))


UNSUPPORTED_LAYERS = [
    NoAttentionLayer,
    CrossAttentionLayer,
    AugAssignLayer,
    InPlaceLayer,
    LoopLayer,
    SplitParallelLayer,
    HelperMethodLayer,
    TwoMlpCallsLayer,
]
