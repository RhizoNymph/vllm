// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeSet, HashMap};

use enum_as_inner::EnumAsInner;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_default::DefaultFromSerde;
use serde_repr::{Deserialize_repr, Serialize_repr};
use serde_tuple::Serialize_tuple;

use super::utility::UtilityOutput;
use crate::error::{Error, Result, ext_value_decode};
use crate::protocol::capture::CaptureResult;
use crate::protocol::logprobs::MaybeWireLogprobs;
use crate::protocol::stats::{PrefillStats, SchedulerStats};
use crate::protocol::{OpaqueValue, decode_msgpack};

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

/// Engine-core output for a single request.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/d3af8c18317c0dc008d42e4367fbb9045cfb7bf6/vllm/v1/engine/__init__.py#L154-L184>
// Decoded with a hand-written seq visitor (not `Deserialize_tuple`) so the
// frontend tolerates BOTH shorter tuples (upstream `omit_defaults` elides
// trailing defaults) AND longer tuples (upstream appends a new `array_like`
// field the frontend does not know yet): unknown trailing elements are drained
// instead of raising a decode error. See the `Deserialize` impl below.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, DefaultFromSerde)]
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
    pub pooling_output: Option<OpaqueValue>,
    #[serde(default)]
    pub finish_reason: Option<EngineCoreFinishReason>,
    #[serde(default)]
    pub stop_reason: Option<StopReason>,
    /// Per-request activation-capture results, keyed by consumer name. Populated
    /// by engine-core for requests that opted into capture; empty otherwise.
    ///
    /// MUST stay positioned between `stop_reason` and `events` to match the
    /// Python `EngineCoreOutput` tuple layout (`array_like`).
    #[serde(default)]
    pub capture_results: HashMap<String, CaptureResult>,
    #[serde(default)]
    pub events: Option<Vec<EngineCoreEvent>>,
    #[serde(default)]
    pub kv_transfer_params: Option<serde_json::Value>,
    #[serde(default)]
    pub ec_transfer_params: Option<serde_json::Value>,
    #[serde(default)]
    pub trace_headers: Option<OpaqueValue>,
    /// Breakdown of the scheduled prefill computation, set on the first output
    /// of a newly scheduled prefill and elided for subsequent decode outputs.
    #[serde(default)]
    pub prefill_stats: Option<PrefillStats>,
    #[serde(default)]
    pub routed_experts: Option<OpaqueValue>,
    /// Number of NaNs seen in logits. Values above zero indicate corruption.
    #[serde(default)]
    pub num_nans_in_logits: u32,
}

impl EngineCoreOutput {
    /// Returns whether this output is terminal for the request.
    pub fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }

    /// Resolve all wire-format fields in-place by looking up aux frames and
    /// decoding raw-view payloads as needed.
    fn resolve_in_place<Frame>(&mut self, frames: &[Frame]) -> Result<()>
    where
        Frame: AsRef<[u8]>,
    {
        self.new_logprobs = (self.new_logprobs.take())
            .map(|value| value.resolve(frames, "new_logprobs"))
            .transpose()?;
        self.new_prompt_logprobs_tensors = (self.new_prompt_logprobs_tensors.take())
            .map(|value| value.resolve(frames, "new_prompt_logprobs_tensors"))
            .transpose()?;
        Ok(())
    }
}

impl<'de> Deserialize<'de> for EngineCoreOutput {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct OutputVisitor;

        impl<'de> serde::de::Visitor<'de> for OutputVisitor {
            type Value = EngineCoreOutput;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an EngineCoreOutput array_like tuple")
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                // Missing trailing elements fall back to defaults (omit_defaults);
                // any elements past the known fields are drained (forward-compat).
                let output = EngineCoreOutput {
                    request_id: seq.next_element()?.unwrap_or_default(),
                    new_token_ids: seq.next_element()?.unwrap_or_default(),
                    new_logprobs: seq.next_element()?.unwrap_or_default(),
                    new_prompt_logprobs_tensors: seq.next_element()?.unwrap_or_default(),
                    pooling_output: seq.next_element()?.unwrap_or_default(),
                    finish_reason: seq.next_element()?.unwrap_or_default(),
                    stop_reason: seq.next_element()?.unwrap_or_default(),
                    capture_results: seq.next_element()?.unwrap_or_default(),
                    events: seq.next_element()?.unwrap_or_default(),
                    kv_transfer_params: seq.next_element()?.unwrap_or_default(),
                    ec_transfer_params: seq.next_element()?.unwrap_or_default(),
                    trace_headers: seq.next_element()?.unwrap_or_default(),
                    prefill_stats: seq.next_element()?.unwrap_or_default(),
                    routed_experts: seq.next_element()?.unwrap_or_default(),
                    num_nans_in_logits: seq.next_element()?.unwrap_or_default(),
                };
                while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {}
                Ok(output)
            }
        }

        deserializer.deserialize_seq(OutputVisitor)
    }
}

/// Raw Python/msgpack engine-core output envelope.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/engine/__init__.py#L186-L214>
// See `EngineCoreOutput`: decoded with a hand-written seq visitor so the
// frontend tolerates both shorter and longer `array_like` tuples across
// upstream wire-format changes.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, DefaultFromSerde)]
struct WireEngineCoreOutputs {
    #[serde(default)]
    engine_index: u32,
    /// Outputs grouped for this client in the current engine tick.
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
                    outputs: seq.next_element()?.unwrap_or_default(),
                    scheduler_stats: seq.next_element()?.unwrap_or_default(),
                    timestamp: seq.next_element()?.unwrap_or_default(),
                    utility_output: seq.next_element()?.unwrap_or_default(),
                    late_capture_results: seq.next_element()?.unwrap_or_default(),
                    finished_requests: seq.next_element()?.unwrap_or_default(),
                    wave_complete: seq.next_element()?.unwrap_or_default(),
                    start_wave: seq.next_element()?.unwrap_or_default(),
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
    /// Resolve all wire-format fields in-place by looking up aux frames and
    /// decoding raw-view payloads as needed.
    fn resolve_in_place<Frame>(&mut self, frames: &[Frame]) -> Result<()>
    where
        Frame: AsRef<[u8]>,
    {
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

/// Decode one ordinary or multipart engine-core output message into the strong
/// typed public protocol shape.
pub fn decode_engine_core_outputs<Frame>(frames: &[Frame]) -> Result<EngineCoreOutputs>
where
    Frame: AsRef<[u8]>,
{
    let first_frame = frames.first().ok_or_else(|| ext_value_decode!("missing output frame"))?;

    let mut outputs: EngineCoreOutputs = decode_msgpack(first_frame.as_ref())?;
    outputs.resolve_in_place(frames)?;
    Ok(outputs)
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
                new_logprobs: None,
                new_prompt_logprobs_tensors: None,
                pooling_output: None,
                finish_reason: Some(EngineCoreFinishReason::Length),
                stop_reason: Some(StopReason::Text("stop".to_string())),
                capture_results: Default::default(),
                events: None,
                kv_transfer_params: None,
                ec_transfer_params: None,
                trace_headers: None,
                prefill_stats: None,
                routed_experts: None,
                num_nans_in_logits: 0,
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
                        },
                    ],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    finished_requests: Some(
                        {
                            "req-1",
                        },
                    ),
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
            Value::from(true),    // hypothetical future trailing field
        ]);

        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &outer).unwrap();
        let decoded: WireEngineCoreOutputs = decode_msgpack(&bytes)
            .expect("an EngineCoreOutputs frame with an unknown trailing field must still decode");
        assert_eq!(decoded.engine_index, 0);
    }
}
