// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use bytes::Bytes;
use enum_as_inner::EnumAsInner;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_default::DefaultFromSerde;
use serde_repr::{Deserialize_repr, Serialize_repr};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};
use serde_with::de::DeserializeAsWrap;

use super::serde_utils::AllowTrailingFields;
use super::utility::UtilityOutput;
use crate::error::{Error, Result, ext_value_decode};
use crate::protocol::capture::CaptureResult;
use crate::protocol::logprobs::MaybeWireLogprobs;
use crate::protocol::sampling_mask::MaybeWireSamplingMask;
use crate::protocol::stats::{PrefillStats, SchedulerStats};
use crate::protocol::tensor::WireNdArray;
use crate::protocol::{OpaqueValue, decode_msgpack, encode_msgpack};

/// The stop reason associated with a finished output.
///
/// Python models this as the union-typed `stop_reason: int | str | None`
/// field on `EngineCoreOutput`; the Rust client narrows it into a tagged enum.
///
/// Original Python field:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L155>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopReason {
    TokenId(u32),
    Text(String),
}

/// Reason a request finished: stop, length, abort, error, or repetition.
///
/// This mirrors the Python enum and uses integer encoding for compact wire
/// representation.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L41-L63>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum EngineCoreFinishReason {
    /// A stop string was emitted.
    Stop = 0,
    /// `max_tokens` or `max_model_len` was reached.
    Length = 1,
    /// The request was aborted by the client.
    Abort = 2,
    /// A retryable request-level internal error occurred.
    Error = 3,
    /// A repetitive token pattern was detected.
    Repetition = 4,
}

/// Event types emitted by engine-core for one request.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L113-L118>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum EngineCoreEventType {
    Queued = 1,
    Scheduled = 2,
    Preempted = 3,
}

/// A timestamped engine-core event associated with one request.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L121-L130>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineCoreEvent {
    pub r#type: EngineCoreEventType,
    pub timestamp: f64,
}

/// Decode a fork-only `array_like` slot, treating `nil` like an omitted
/// value: upstream-shaped senders put their own `nil` defaults at the index the
/// fork's field occupies.
fn nil_as_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// Engine-core output for a single request.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/d3af8c18317c0dc008d42e4367fbb9045cfb7bf6/vllm/v1/engine/__init__.py#L154-L184>
///
/// Fields that are set on few outputs (first or final output, or opt-in
/// features) are boxed: outputs are moved by value through the dispatcher and
/// the request stream once per engine step, so their inline size matters.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct EngineCoreOutput {
    pub request_id: String,
    pub new_token_ids: Vec<u32>,
    /// Decoded sample logprobs for the newly generated positions in this
    /// output.
    #[serde(default)]
    pub new_logprobs: Option<MaybeWireLogprobs>,
    /// Decoded prompt logprobs for the scored prompt positions emitted in this
    /// output.
    #[serde(default)]
    pub new_prompt_logprobs_tensors: Option<MaybeWireLogprobs>,
    #[serde(default)]
    pub pooling_output: Option<Box<OpaqueValue>>,
    #[serde(default)]
    pub finish_reason: Option<EngineCoreFinishReason>,
    #[serde(default)]
    pub stop_reason: Option<StopReason>,
    /// Per-request activation-capture results, keyed by consumer name. Populated
    /// by engine-core for requests that opted into capture; empty otherwise.
    ///
    /// MUST stay positioned between `stop_reason` and `events` to match the
    /// Python `EngineCoreOutput` tuple layout (`array_like`). A `nil` slot
    /// (a sender that does not know the fork field) decodes as empty.
    #[serde(default, deserialize_with = "nil_as_default")]
    pub capture_results: HashMap<String, CaptureResult>,
    #[serde(default)]
    pub events: Option<Vec<EngineCoreEvent>>,
    #[serde(default)]
    pub kv_transfer_params: Option<Box<serde_json::Value>>,
    #[serde(default)]
    pub ec_transfer_params: Option<Box<serde_json::Value>>,
    #[serde(default)]
    pub trace_headers: Option<Box<OpaqueValue>>,
    /// Breakdown of the scheduled prefill computation, set on the first output
    /// of a newly scheduled prefill and elided for subsequent decode outputs.
    #[serde(default)]
    pub prefill_stats: Option<Box<PrefillStats>>,
    /// Routed-expert rows `[num_rows, num_layers, top_k]` captured for this
    /// output, when `enable_return_routed_experts` is on.
    #[serde(default)]
    pub routed_experts: Option<Box<WireNdArray>>,
    /// Number of NaNs seen in logits. Values above zero indicate corruption.
    #[serde(default)]
    pub num_nans_in_logits: u32,
    /// Multi-modal hashes the engine could not find in its receiver cache,
    /// because the frontend sent `data: None` for an item the engine had
    /// already evicted. Non-empty makes the output retryable: a frontend that
    /// keeps a metadata-only shadow of the engine cache drops these entries and
    /// resends the request with the item data attached.
    ///
    /// TODO: always `None` here, since this client has no multi-modal processor
    /// cache and always sends item data inline. Act on it once the Rust
    /// frontend grows one.
    #[serde(default)]
    pub mm_cache_miss_hashes: Option<Vec<String>>,
    #[serde(default)]
    pub new_sampling_mask: Option<MaybeWireSamplingMask>,
    /// Per-request speculative-decoding acceptance metrics, set on the final
    /// output when `--per-request-spec-decode-metrics` is enabled.
    #[serde(default)]
    pub spec_decode_metrics: Option<Box<RequestSpecDecodeMetrics>>,
    /// Log probabilities of `SamplingParams.prompt_logprob_token_ids`, set on
    /// the first output of a request that asked for them.
    #[serde(default)]
    pub prompt_token_id_logprobs: Option<Box<WireNdArray>>,
    /// Request-level error sent with [`EngineCoreFinishReason::Error`], e.g.
    /// when a paused engine rejects a request.
    #[serde(default)]
    pub error: Option<Box<RequestError>>,
    /// Whether the sampler watermarks this request, after EngineCore applied
    /// `watermark_config.enforce`. Set on the first output, and only when the
    /// engine has a `watermark_config`.
    #[serde(default)]
    pub watermarking: Option<bool>,
}

/// A request-level error that does not mark the engine as unhealthy.
///
/// Python models this as a plain `@dataclass`, so it is serialized by msgspec
/// as a map (named fields).
///
/// Original Python definition: `RequestError` in `vllm/outputs.py`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestError {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub retryable: bool,
}

impl RequestError {
    /// `code` of an error caused by the request itself, e.g. a structured-output
    /// constraint the engine's backend cannot compile. Python raises
    /// `VLLMValidationError` for it.
    pub const VALIDATION_ERROR: &str = "validation_error";

    /// Whether the request itself is invalid, so frontends report a client
    /// error.
    pub fn is_validation_error(&self) -> bool {
        self.code == Self::VALIDATION_ERROR
    }
}

/// Raw per-sequence speculative-decoding accumulator.
///
/// Python models this as a plain `@dataclass`, so it is serialized by msgspec
/// as a map (named fields). Derived values such as acceptance rates are
/// computed by the frontend, not carried on the wire.
///
/// Original Python definition: `RequestSpecDecodeMetrics` in
/// `vllm/v1/metrics/stats.py`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestSpecDecodeMetrics {
    /// Configured `num_speculative_tokens`.
    #[serde(default)]
    pub num_spec_tokens: u64,
    /// Verify-step counts indexed by accepted draft-token count.
    #[serde(default)]
    pub histogram: Vec<u64>,
    /// Total proposed draft tokens.
    #[serde(default)]
    pub num_draft_tokens: u64,
    /// Accepted draft count per verify step; empty unless `detailed`.
    #[serde(default)]
    pub per_step_accepted: Vec<u64>,
    /// Proposed draft count per verify step; empty unless `detailed`.
    #[serde(default)]
    pub per_step_drafted: Vec<u64>,
    /// Total draft tokens the target model ran a forward pass on.
    #[serde(default)]
    pub num_verified_tokens: u64,
    /// Verified draft count per verify step; empty unless `detailed`.
    #[serde(default)]
    pub per_step_verified: Vec<u64>,
}

impl EngineCoreOutput {
    /// Returns whether this output is terminal for the request.
    pub fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }

    /// Move the large arrays of wire-format fields into the ordered
    /// auxiliary-frame list, the inverse of [`Self::resolve_in_place`].
    fn extract_aux_frames(&mut self, aux_frames: &mut Vec<Bytes>, threshold: usize) {
        if let Some(value) = &mut self.new_logprobs {
            value.extract_aux_frames(aux_frames, threshold);
        }
        if let Some(value) = &mut self.new_prompt_logprobs_tensors {
            value.extract_aux_frames(aux_frames, threshold);
        }
        if let Some(value) = &mut self.new_sampling_mask {
            value.extract_aux_frames(aux_frames, threshold);
        }
        if let Some(value) = &mut self.prompt_token_id_logprobs {
            value.extract_aux_frame(aux_frames, threshold);
        }
        if let Some(value) = &mut self.routed_experts {
            value.extract_aux_frame(aux_frames, threshold);
        }
    }

    /// Resolve all wire-format fields in-place by looking up aux frames and
    /// decoding raw-view payloads as needed.
    fn resolve_in_place(&mut self, frames: &[Bytes]) -> Result<()> {
        self.new_logprobs = (self.new_logprobs.take())
            .map(|value| value.resolve(frames, "new_logprobs"))
            .transpose()?;
        self.new_prompt_logprobs_tensors = (self.new_prompt_logprobs_tensors.take())
            .map(|value| value.resolve(frames, "new_prompt_logprobs_tensors"))
            .transpose()?;
        self.new_sampling_mask = (self.new_sampling_mask.take())
            .map(|value| value.resolve(frames, "new_sampling_mask"))
            .transpose()?;
        if let Some(value) = self.prompt_token_id_logprobs.as_mut() {
            value
                .resolve_aux_frame(frames)
                .map_err(|message| ext_value_decode!("prompt_token_id_logprobs: {message}"))?;
        }
        if let Some(value) = self.routed_experts.as_mut() {
            value
                .resolve_aux_frame(frames)
                .map_err(|message| ext_value_decode!("routed_experts: {message}"))?;
        }
        Ok(())
    }
}

/// Raw Python/msgpack engine-core output envelope.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L186-L214>
// Decoded with a hand-written seq visitor (not `Deserialize_tuple`) so the
// frontend tolerates BOTH shorter tuples (upstream `omit_defaults` elides
// trailing defaults) AND longer tuples (upstream appends a new `array_like`
// envelope field the frontend does not know yet): unknown trailing elements
// are drained instead of raising a decode error. Each element of `outputs` is
// decoded with `AllowTrailingFields` for the same reason. See the
// `Deserialize` impl below; its field order MUST match this struct's.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, DefaultFromSerde)]
struct WireEngineCoreOutputs {
    #[serde(default)]
    engine_index: u32,
    /// Outputs grouped for this client in the current engine tick. Each is
    /// decoded with `AllowTrailingFields` to match msgspec's append-only
    /// schema evolution, including Omni extensions.
    #[serde(default)]
    outputs: Vec<EngineCoreOutput>,
    #[serde(default)]
    scheduler_stats: Option<Box<SchedulerStats>>,
    #[serde(default)]
    timestamp: f64,
    #[serde(default)]
    utility_output: Option<UtilityOutput>,
    /// Capture results that finalized AFTER their request finished (writes are
    /// asynchronous). Keyed by `request_id`, then by consumer name. Not yet
    /// routed to `capture_wait` waiters, but the field MUST be present so the
    /// `array_like` tuple stays aligned with the Python `EngineCoreOutputs` wire
    /// layout (it sits between `utility_output` and `finished_requests`).
    #[serde(default)]
    late_capture_results: HashMap<String, HashMap<String, CaptureResult>>,
    #[serde(default)]
    finished_requests: Option<BTreeSet<String>>,
    /// In DP mode, signals that the current wave finished and engines are
    /// paused.
    #[serde(default)]
    wave_complete: Option<u32>,
    /// In DP mode, signals that a request arrived for an old wave and the next
    /// wave needs to start in other engines.
    #[serde(default)]
    start_wave: Option<u32>,
    /// Label of the weights used for this step, as set by the
    /// `set_weight_version` utility; `"default"` until set.
    #[serde(default)]
    weight_version: Option<String>,
}

impl<'de> Deserialize<'de> for WireEngineCoreOutputs {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct OutputsVisitor;

        impl<'de> serde::de::Visitor<'de> for OutputsVisitor {
            type Value = WireEngineCoreOutputs;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an EngineCoreOutputs array_like tuple")
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                // Missing trailing elements fall back to defaults (omit_defaults);
                // any elements past the known fields are drained (forward-compat).
                let outputs = WireEngineCoreOutputs {
                    engine_index: seq.next_element()?.unwrap_or_default(),
                    outputs: seq
                        .next_element::<DeserializeAsWrap<
                            Vec<EngineCoreOutput>,
                            Vec<AllowTrailingFields>,
                        >>()?
                        .map(DeserializeAsWrap::into_inner)
                        .unwrap_or_default(),
                    scheduler_stats: seq.next_element()?.unwrap_or_default(),
                    timestamp: seq.next_element()?.unwrap_or_default(),
                    utility_output: seq.next_element()?.unwrap_or_default(),
                    // `nil` here (a sender without the fork field) is empty.
                    late_capture_results: seq
                        .next_element::<Option<_>>()?
                        .flatten()
                        .unwrap_or_default(),
                    finished_requests: seq.next_element()?.unwrap_or_default(),
                    wave_complete: seq.next_element()?.unwrap_or_default(),
                    start_wave: seq.next_element()?.unwrap_or_default(),
                    weight_version: seq.next_element()?.unwrap_or_default(),
                };
                while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {}
                Ok(outputs)
            }
        }

        deserializer.deserialize_seq(OutputsVisitor)
    }
}

/// Data-parallel control notifications multiplexed through `EngineCoreOutputs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DpControlMessage {
    WaveComplete(u32),
    StartWave(u32),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RequestBatchOutputs {
    pub engine_index: u32,
    pub outputs: Vec<EngineCoreOutput>,
    pub scheduler_stats: Option<Box<SchedulerStats>>,
    pub timestamp: f64,
    pub finished_requests: Option<BTreeSet<String>>,
    /// Label of the weights used for this step. It changes only while
    /// generation is paused, so it applies to every output in the batch.
    pub weight_version: Option<Arc<str>>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct UtilityCallOutput {
    pub engine_index: u32,
    pub timestamp: f64,
    pub output: UtilityOutput,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DpControlOutput {
    pub engine_index: u32,
    pub timestamp: f64,
    pub control: DpControlMessage,
}

/// Semantic engine-core output families.
///
/// Python currently uses one product-shaped wire struct. The Rust protocol
/// exposes the finite semantic families while preserving the same msgpack shape
/// for serialization.
#[derive(Debug, Clone, PartialEq, EnumAsInner)]
pub enum EngineCoreOutputs {
    RequestBatch(RequestBatchOutputs),
    Utility(UtilityCallOutput),
    DpControl(DpControlOutput),
}

impl From<RequestBatchOutputs> for EngineCoreOutputs {
    fn from(outputs: RequestBatchOutputs) -> Self {
        Self::RequestBatch(outputs)
    }
}

impl From<UtilityCallOutput> for EngineCoreOutputs {
    fn from(output: UtilityCallOutput) -> Self {
        Self::Utility(output)
    }
}

impl From<DpControlOutput> for EngineCoreOutputs {
    fn from(output: DpControlOutput) -> Self {
        Self::DpControl(output)
    }
}

impl EngineCoreOutputs {
    /// Move the large arrays of wire-format fields into the ordered
    /// auxiliary-frame list, the inverse of [`Self::resolve_in_place`].
    fn extract_aux_frames(&mut self, threshold: usize) -> Vec<Bytes> {
        let mut aux_frames = Vec::new();
        if let Self::RequestBatch(batch) = self {
            for output in &mut batch.outputs {
                output.extract_aux_frames(&mut aux_frames, threshold);
            }
        }
        aux_frames
    }

    /// Resolve all wire-format fields in-place by looking up aux frames and
    /// decoding raw-view payloads as needed.
    fn resolve_in_place(&mut self, frames: &[Bytes]) -> Result<()> {
        if let Self::RequestBatch(batch) = self {
            for output in &mut batch.outputs {
                output.resolve_in_place(frames)?;
            }
        }
        Ok(())
    }
}

/// Classify the raw wire message into a more semantic Rust enum.
impl TryFrom<WireEngineCoreOutputs> for EngineCoreOutputs {
    type Error = Error;

    fn try_from(value: WireEngineCoreOutputs) -> Result<Self> {
        let has_request_payload = !value.outputs.is_empty()
            || value.scheduler_stats.is_some()
            || value.finished_requests.is_some()
            || !value.late_capture_results.is_empty();

        match (
            has_request_payload,
            &value.utility_output,
            &value.wave_complete,
            &value.start_wave,
        ) {
            (true, None, None, None) => Ok(RequestBatchOutputs {
                engine_index: value.engine_index,
                outputs: value.outputs,
                scheduler_stats: value.scheduler_stats,
                timestamp: value.timestamp,
                finished_requests: value.finished_requests,
                weight_version: value.weight_version.map(Arc::from),
            }
            .into()),
            (false, Some(_), None, None) => Ok(UtilityCallOutput {
                engine_index: value.engine_index,
                timestamp: value.timestamp,
                output: value.utility_output.unwrap(),
            }
            .into()),
            (false, None, Some(_), None) => Ok(DpControlOutput {
                engine_index: value.engine_index,
                timestamp: value.timestamp,
                control: DpControlMessage::WaveComplete(value.wave_complete.unwrap()),
            }
            .into()),
            (false, None, None, Some(_)) => Ok(DpControlOutput {
                engine_index: value.engine_index,
                timestamp: value.timestamp,
                control: DpControlMessage::StartWave(value.start_wave.unwrap()),
            }
            .into()),

            _ => Err(Error::Decode {
                target_type: "EngineCoreOutputs",
                message: "invalid wire shape".to_string(),
            }),
        }
    }
}

impl From<EngineCoreOutputs> for WireEngineCoreOutputs {
    fn from(value: EngineCoreOutputs) -> Self {
        match value {
            EngineCoreOutputs::RequestBatch(batch) => Self {
                engine_index: batch.engine_index,
                outputs: batch.outputs,
                scheduler_stats: batch.scheduler_stats,
                timestamp: batch.timestamp,
                finished_requests: batch.finished_requests,
                weight_version: batch.weight_version.map(|version| version.to_string()),
                ..Default::default()
            },
            EngineCoreOutputs::Utility(utility) => Self {
                engine_index: utility.engine_index,
                timestamp: utility.timestamp,
                utility_output: Some(utility.output),
                ..Default::default()
            },
            EngineCoreOutputs::DpControl(control) => {
                let (wave_complete, start_wave) = match control.control {
                    DpControlMessage::WaveComplete(wave) => (Some(wave), None),
                    DpControlMessage::StartWave(wave) => (None, Some(wave)),
                };
                Self {
                    engine_index: control.engine_index,
                    timestamp: control.timestamp,
                    wave_complete,
                    start_wave,
                    ..Default::default()
                }
            }
        }
    }
}

impl Serialize for EngineCoreOutputs {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Clones the batch; an engine encodes by value with `into_frames`
        // instead.
        WireEngineCoreOutputs::from(self.clone()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EngineCoreOutputs {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        WireEngineCoreOutputs::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

impl EngineCoreOutputs {
    /// Decode one ordinary or multipart engine-core output message into the strong
    /// typed public protocol shape.
    pub fn from_frames(frames: &[Bytes]) -> Result<Self> {
        let first_frame =
            frames.first().ok_or_else(|| ext_value_decode!("missing output frame"))?;

        let mut outputs: Self = decode_msgpack(first_frame.as_ref())?;
        outputs.resolve_in_place(frames)?;
        Ok(outputs)
    }

    /// Encode into one ordinary or multipart message: the msgpack frame, then
    /// an auxiliary frame for each array of at least `aux_frame_threshold`
    /// bytes, as Python's `MsgpackEncoder` does. The inverse of
    /// [`Self::from_frames`].
    pub fn into_frames(mut self, aux_frame_threshold: usize) -> Result<Vec<Bytes>> {
        let aux_frames = self.extract_aux_frames(aux_frame_threshold);
        let mut frames = Vec::with_capacity(1 + aux_frames.len());
        frames.push(Bytes::from(encode_msgpack(&WireEngineCoreOutputs::from(
            self,
        ))?));
        frames.extend(aux_frames);
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use rmpv::Value;

    use super::*;
    use crate::protocol::output::EngineCoreOutput;
    use crate::protocol::{decode_msgpack, decode_value, encode_msgpack};

    #[test]
    fn engine_core_outputs_roundtrip_finished_fields() {
        let outputs = WireEngineCoreOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![42],
                finish_reason: Some(EngineCoreFinishReason::Length),
                stop_reason: Some(StopReason::Text("stop".to_string())),
                ..Default::default()
            }],
            finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
            ..Default::default()
        };

        let encoded = encode_msgpack(&outputs).unwrap();
        let decoded: WireEngineCoreOutputs = decode_msgpack(&encoded).unwrap();

        assert_eq!(decoded.outputs.len(), 1);
        assert_eq!(
            decoded.outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Length)
        );
        assert_eq!(
            decoded.finished_requests,
            Some(BTreeSet::from(["req-1".to_string()]))
        );
    }

    #[test]
    fn engine_core_outputs_ignore_appended_fields() {
        let output = EngineCoreOutput {
            request_id: "req-1".into(),
            new_token_ids: vec![42],
            finish_reason: Some(EngineCoreFinishReason::Length),
            ..Default::default()
        };
        let mut fields = crate::protocol::decode_value(&encode_msgpack(&output).unwrap())
            .unwrap()
            .as_array()
            .unwrap()
            .clone();
        // Omni appends multimodal_output, is_segment_finished, and
        // new_prompt_len_snapshot. Unknown payloads may contain nested values.
        fields.extend([
            OpaqueValue::Map(vec![(
                "audio".into(),
                OpaqueValue::Array(vec![OpaqueValue::Ext(1, vec![0, 1])]),
            )]),
            false.into(),
            OpaqueValue::Nil,
        ]);
        let wire = (0, vec![OpaqueValue::Array(fields); 2]);
        let frames = [Bytes::from(encode_msgpack(&wire).unwrap())];
        let decoded = EngineCoreOutputs::from_frames(&frames).unwrap();
        // A second output also checks that the first output's tail was consumed.
        assert_eq!(decoded.as_request_batch().unwrap().outputs, vec![output; 2]);
    }

    #[test]
    fn engine_core_output_requires_valid_known_fields() {
        for fields in [
            vec![],
            vec!["req-1".into()],
            vec!["req-1".into(), OpaqueValue::Nil],
            vec!["req-1".into(), OpaqueValue::Array(vec![]), false.into()],
        ] {
            let wire = (0, vec![OpaqueValue::Array(fields)]);
            let frames = [Bytes::from(encode_msgpack(&wire).unwrap())];
            assert!(EngineCoreOutputs::from_frames(&frames).is_err());
        }
        let wire = (0, vec![("req-1", vec![42_u32])]);
        let frames = [Bytes::from(encode_msgpack(&wire).unwrap())];
        let decoded = EngineCoreOutputs::from_frames(&frames).unwrap();
        assert_eq!(
            decoded.as_request_batch().unwrap().outputs,
            vec![EngineCoreOutput {
                request_id: "req-1".into(),
                new_token_ids: vec![42],
                ..Default::default()
            }]
        );
    }

    #[test]
    fn engine_core_outputs_classify_request_batch() {
        let outputs = WireEngineCoreOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![7],
                ..Default::default()
            }],
            finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
            ..Default::default()
        };

        expect_test::expect![[r#"
            RequestBatch(
                RequestBatchOutputs {
                    engine_index: 0,
                    outputs: [
                        EngineCoreOutput {
                            request_id: "req-1",
                            new_token_ids: [
                                7,
                            ],
                            new_logprobs: None,
                            new_prompt_logprobs_tensors: None,
                            pooling_output: None,
                            finish_reason: None,
                            stop_reason: None,
                            capture_results: {},
                            events: None,
                            kv_transfer_params: None,
                            ec_transfer_params: None,
                            trace_headers: None,
                            prefill_stats: None,
                            routed_experts: None,
                            num_nans_in_logits: 0,
                            mm_cache_miss_hashes: None,
                            new_sampling_mask: None,
                            spec_decode_metrics: None,
                            prompt_token_id_logprobs: None,
                            error: None,
                            watermarking: None,
                        },
                    ],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    finished_requests: Some(
                        {
                            "req-1",
                        },
                    ),
                    weight_version: None,
                },
            )
        "#]]
        .assert_debug_eq(&EngineCoreOutputs::try_from(outputs).unwrap());
    }

    #[test]
    fn engine_core_outputs_classify_utility() {
        let outputs = WireEngineCoreOutputs {
            utility_output: Some(UtilityOutput {
                call_id: 42_u64.into(),
                failure_message: None,
                result: None,
            }),
            ..Default::default()
        };

        expect_test::expect![[r#"
            Utility(
                UtilityCallOutput {
                    engine_index: 0,
                    timestamp: 0.0,
                    output: UtilityOutput {
                        call_id: 42,
                        failure_message: None,
                        result: None,
                    },
                },
            )
        "#]]
        .assert_debug_eq(&EngineCoreOutputs::try_from(outputs).unwrap());
    }

    #[test]
    fn engine_core_outputs_classify_control() {
        let outputs = WireEngineCoreOutputs {
            start_wave: Some(3),
            ..Default::default()
        };

        expect_test::expect![[r#"
            DpControl(
                DpControlOutput {
                    engine_index: 0,
                    timestamp: 0.0,
                    control: StartWave(
                        3,
                    ),
                },
            )
        "#]]
        .assert_debug_eq(&EngineCoreOutputs::try_from(outputs).unwrap());
    }

    #[test]
    fn engine_core_outputs_encode_large_arrays_in_aux_frames() {
        use crate::protocol::logprobs::{Logprobs, PositionLogprobs, TokenLogprob, WireLogprobs};
        use crate::protocol::tensor::WireNdArray;

        const THRESHOLD: usize = 64;
        // One position with 16 candidates: 16 * 8 bytes of token ids go in an
        // auxiliary frame, 16 * 4 bytes of logprobs too, the rank stays inline.
        let entries: Vec<TokenLogprob> = (0..16)
            .map(|i| TokenLogprob {
                token_id: 100 + i,
                logprob: -(i as f32),
                rank: if i == 0 { 3 } else { i },
            })
            .collect();
        let wire = WireLogprobs {
            logprob_token_ids: WireNdArray::from_i64(
                vec![1, 16],
                entries.iter().map(|e| i64::from(e.token_id)).collect(),
            )
            .unwrap(),
            logprobs: WireNdArray::from_f32(
                vec![1, 16],
                entries.iter().map(|e| e.logprob).collect(),
            )
            .unwrap(),
            token_ranks: WireNdArray::from_i64(vec![1], vec![3]).unwrap(),
            cu_num_generated_tokens: None,
            cu_num_generated_tokens_tensor: None,
        };
        let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 1,
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".into(),
                new_token_ids: vec![100],
                new_logprobs: Some(MaybeWireLogprobs::Wire(Box::new(wire))),
                ..Default::default()
            }],
            timestamp: 2.5,
            weight_version: Some("v1".into()),
            ..Default::default()
        });

        let frames = outputs.into_frames(THRESHOLD).unwrap();
        assert_eq!(frames.len(), 3);
        let decoded = EngineCoreOutputs::from_frames(&frames).unwrap();
        let batch = decoded.as_request_batch().unwrap();
        assert_eq!((batch.engine_index, batch.timestamp), (1, 2.5));
        assert_eq!(batch.weight_version.as_deref(), Some("v1"));
        assert_eq!(
            *batch.outputs[0].new_logprobs.as_ref().unwrap().as_direct().unwrap(),
            Logprobs {
                positions: vec![PositionLogprobs { entries }]
            }
        );
    }

    #[test]
    fn engine_core_outputs_encode_utility_and_control() {
        for outputs in [
            EngineCoreOutputs::Utility(UtilityCallOutput {
                engine_index: 2,
                timestamp: 1.0,
                output: UtilityOutput {
                    call_id: 7_u64.into(),
                    failure_message: Some("boom".into()),
                    result: None,
                },
            }),
            EngineCoreOutputs::DpControl(DpControlOutput {
                engine_index: 0,
                timestamp: 1.0,
                control: DpControlMessage::WaveComplete(4),
            }),
        ] {
            let frames = outputs.clone().into_frames(0).unwrap();
            assert_eq!(EngineCoreOutputs::from_frames(&frames).unwrap(), outputs);
        }
    }

    #[test]
    fn engine_core_outputs_rejects_mixed_shape() {
        let outputs = WireEngineCoreOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![7],
                ..Default::default()
            }],
            utility_output: Some(UtilityOutput {
                call_id: 1_u64.into(),
                failure_message: None,
                result: None,
            }),
            ..Default::default()
        };

        let error = EngineCoreOutputs::try_from(outputs).unwrap_err();
        expect_test::expect![[
            r#"messagepack decode failed for EngineCoreOutputs: invalid wire shape"#
        ]]
        .assert_eq(&error.to_string());
    }

    #[test]
    fn engine_core_output_capture_results_sit_at_tuple_index_7() {
        let output = EngineCoreOutput {
            request_id: "r".to_string(),
            new_token_ids: vec![1],
            stop_reason: Some(StopReason::Text("x".to_string())),
            capture_results: HashMap::from([(
                "filesystem".to_string(),
                CaptureResult {
                    key: None,
                    status: "ok".to_string(),
                    error: None,
                    payload: Some(serde_json::json!({ "paths": ["/a.bin"] })),
                },
            )]),
            events: Some(vec![EngineCoreEvent {
                r#type: EngineCoreEventType::Queued,
                timestamp: 1.0,
            }]),
            num_nans_in_logits: 3,
            ..EngineCoreOutput::default()
        };

        // The output is an `array_like` tuple; capture_results MUST land at index
        // 7 (between stop_reason and events) to match Python's layout.
        let encoded = encode_msgpack(&output).unwrap();
        let array = match decode_value(&encoded).unwrap() {
            Value::Array(array) => array,
            other => panic!("expected array, got {other:?}"),
        };
        assert!(
            array[7].is_map(),
            "capture_results must be at index 7, got {:?}",
            array[7]
        );

        // Round-trip preserves capture_results AND every field after it — proving
        // the tuple stays aligned for capture-enabled outputs.
        let decoded: EngineCoreOutput = decode_msgpack(&encoded).unwrap();
        assert_eq!(decoded.capture_results["filesystem"].status, "ok");
        assert!(decoded.events.is_some());
        assert_eq!(decoded.num_nans_in_logits, 3);
    }

    /// Reproduces the real engine-core `EngineCoreOutputs` wire frame captured
    /// live from the Python engine. The regression this pins: the outer tuple
    /// carries `late_capture_results` (an `array_like` field appended upstream)
    /// as an empty map at index 5, between `utility_output` and
    /// `finished_requests`. Before the field existed on the Rust side, that map
    /// was decoded against `finished_requests: Option<set>` and failed with
    /// "invalid type: map, expected a sequence", wedging the whole client.
    ///
    /// The inner `EngineCoreOutput` also mirrors the live shape: `events` is a
    /// list of MAPS (plain msgspec Struct), `prefill_stats` is a MAP
    /// (dataclass), `capture_results` is an empty map, and the trailing
    /// `EngineCoreOutputs` fields (`wave_complete`, `start_wave`) are omitted
    /// (`omit_defaults`), so the sequence is shorter than the full field count.
    #[test]
    fn decodes_real_engine_core_outputs_frame_with_late_capture_results() {
        // Inner EngineCoreOutput as an array_like tuple (15 slots), matching the
        // live plain-completion dump.
        let event = |ty: u8, ts: f64| {
            Value::Map(vec![
                (Value::from("type"), Value::from(ty)),
                (Value::from("timestamp"), Value::from(ts)),
            ])
        };
        let prefill_stats = Value::Map(vec![
            (Value::from("num_prompt_tokens"), Value::from(5u32)),
            (Value::from("num_computed_tokens"), Value::from(5u32)),
            (Value::from("num_cached_tokens"), Value::from(0u32)),
            (Value::from("num_local_cached_tokens"), Value::from(0u32)),
            (Value::from("num_external_cached_tokens"), Value::from(0u32)),
        ]);
        let inner = Value::Array(vec![
            Value::from("cmpl-fcdb0f49-f04b1c4e"),     // request_id
            Value::Array(vec![Value::from(12095u32)]), // new_token_ids
            Value::Nil,                                // new_logprobs
            Value::Nil,                                // new_prompt_logprobs_tensors
            Value::Nil,                                // pooling_output
            Value::from(1u8),                          // finish_reason (Length)
            Value::Nil,                                // stop_reason
            Value::Map(vec![]),                        // capture_results (empty map)
            Value::Array(vec![event(1, 1495680.63), event(2, 1495680.63)]), // events (list of maps)
            Value::Nil,                                // kv_transfer_params
            Value::Nil,                                // ec_transfer_params
            Value::Nil,                                // trace_headers
            prefill_stats,                             // prefill_stats (map)
            Value::Nil,                                // routed_experts
            Value::from(0u32),                         // num_nans_in_logits
        ]);

        // Real scheduler_stats is a dataclass -> named map with every field
        // present; reuse the Rust encoder to produce that exact map shape.
        let scheduler_stats =
            decode_value(&encode_msgpack(&SchedulerStats::default()).unwrap()).unwrap();
        assert!(scheduler_stats.is_map());

        // Outer EngineCoreOutputs: only the first 7 slots are present; the
        // trailing wave_complete/start_wave are elided by omit_defaults.
        let outer = Value::Array(vec![
            Value::from(0u32),                 // engine_index
            Value::Array(vec![inner]),         // outputs
            scheduler_stats,                   // scheduler_stats (map)
            Value::from(1495680.686869659f64), // timestamp
            Value::Nil,                        // utility_output
            Value::Map(vec![]),                // late_capture_results (empty map) <- regression
            Value::Nil,                        // finished_requests
        ]);

        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &outer).unwrap();

        let decoded: WireEngineCoreOutputs = decode_msgpack(&bytes)
            .expect("a real engine-core outputs frame (late_capture_results map) must decode");

        assert_eq!(decoded.engine_index, 0);
        assert_eq!(decoded.outputs.len(), 1);
        let output = &decoded.outputs[0];
        assert_eq!(output.request_id, "cmpl-fcdb0f49-f04b1c4e");
        assert_eq!(output.new_token_ids, vec![12095]);
        assert_eq!(output.finish_reason, Some(EngineCoreFinishReason::Length));
        assert_eq!(output.events.as_ref().map(|events| events.len()), Some(2));
        assert_eq!(
            output.events.as_ref().unwrap()[1].r#type,
            EngineCoreEventType::Scheduled
        );
        assert_eq!(output.prefill_stats.as_ref().unwrap().num_prompt_tokens, 5);
        assert!(decoded.late_capture_results.is_empty());
        assert!(decoded.scheduler_stats.is_some());
        // Elided trailing fields fall back to their defaults.
        assert_eq!(decoded.wave_complete, None);
        assert_eq!(decoded.start_wave, None);
    }

    /// A non-empty `late_capture_results` map decodes into the nested
    /// `{request_id: {consumer: CaptureResult}}` form and keeps the tuple
    /// aligned so `finished_requests` still decodes after it.
    #[test]
    fn decodes_engine_core_outputs_with_populated_late_capture_results() {
        let capture_result = Value::Map(vec![
            (Value::from("status"), Value::from("ok")),
            (
                Value::from("payload"),
                Value::Map(vec![(
                    Value::from("paths"),
                    Value::Array(vec![Value::from("/tmp/a.bin")]),
                )]),
            ),
        ]);
        let late = Value::Map(vec![(
            Value::from("cmpl-late"),
            Value::Map(vec![(Value::from("filesystem"), capture_result)]),
        )]);
        let outer = Value::Array(vec![
            Value::from(0u32),                            // engine_index
            Value::Array(vec![]),                         // outputs
            Value::Nil,                                   // scheduler_stats
            Value::from(1.0f64),                          // timestamp
            Value::Nil,                                   // utility_output
            late,                                         // late_capture_results (populated)
            Value::Array(vec![Value::from("cmpl-late")]), // finished_requests
        ]);

        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &outer).unwrap();
        let decoded: WireEngineCoreOutputs = decode_msgpack(&bytes).unwrap();

        assert_eq!(
            decoded.late_capture_results["cmpl-late"]["filesystem"].status,
            "ok"
        );
        assert_eq!(
            decoded.finished_requests,
            Some(BTreeSet::from(["cmpl-late".to_string()]))
        );
    }

    /// Forward-compatibility: if upstream appends a NEW trailing field to the
    /// `array_like` `EngineCoreOutputs` tuple, the Rust decoder must not break.
    /// A sequence LONGER than the known field count decodes by ignoring the
    /// extra trailing element(s).
    #[test]
    fn decodes_engine_core_outputs_with_extra_trailing_field() {
        let outer = Value::Array(vec![
            Value::from(0u32),    // engine_index
            Value::Array(vec![]), // outputs
            Value::Nil,           // scheduler_stats
            Value::from(1.0f64),  // timestamp
            Value::Nil,           // utility_output
            Value::Map(vec![]),   // late_capture_results
            Value::Nil,           // finished_requests
            Value::Nil,           // wave_complete
            Value::Nil,           // start_wave
            Value::Nil,           // weight_version
            Value::from(true),    // hypothetical future trailing field
        ]);

        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &outer).unwrap();
        let decoded: WireEngineCoreOutputs = decode_msgpack(&bytes)
            .expect("an EngineCoreOutputs frame with an unknown trailing field must still decode");
        assert_eq!(decoded.engine_index, 0);
    }
}
