// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;

use crate::protocol::clamps::SteeringClamps;
use crate::protocol::steering::SteeringVectorSpec;
use crate::protocol::structured_outputs::StructuredOutputsParams;

fn default_top_p() -> f32 {
    1.0
}

fn default_repetition_penalty() -> f32 {
    1.0
}

fn default_temperature() -> f32 {
    1.0
}

fn default_max_tokens() -> u32 {
    16
}

///
/// Parameters for detecting repetitive N-gram patterns in output tokens.
///
/// Mirrors Python's `RepetitionDetectionParams`:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/sampling_params.py#L109-L144>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepetitionDetectionParams {
    /// Maximum N-gram size to check. 0 disables detection.
    pub max_pattern_size: u32,
    /// Minimum N-gram size to check. Defaults to 1 when zero.
    #[serde(default)]
    pub min_pattern_size: u32,
    /// Minimum number of repetitions to trigger detection (must be >= 2).
    pub min_count: u32,
}

impl RepetitionDetectionParams {
    /// Return `true` when the params are effectively disabled (max_pattern_size
    /// is 0).
    pub fn is_disabled(&self) -> bool {
        self.max_pattern_size == 0
    }
}

/// Engine-core-facing sampling parameters for text generation.
///
/// This is the normalized southbound subset used by the Rust frontend when it
/// talks to Python engine-core over the wire. User-facing request semantics
/// such as `stop` strings, `n`, `ignore_eos`, and output aggregation mode are
/// intentionally handled by higher layers before values reach this DTO.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/sampling_params.py#L155-L291>
// Python's SamplingParams is `omit_defaults=True`, so msgpack drops
// default-valued keys; default the whole struct. Per-field fns cover the
// non-zero defaults.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, DefaultFromSerde)]
#[serde(default)]
pub struct EngineCoreSamplingParams {
    /// Controls randomness. Lower values are more deterministic; zero means
    /// greedy sampling.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Cumulative probability threshold for nucleus sampling.
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Maximum number of top tokens to consider. `0` means all tokens.
    pub top_k: u32,
    /// Random seed used by the sampler when present.
    pub seed: Option<i64>,
    /// Maximum number of tokens to generate per output sequence.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Minimum number of tokens to generate before EOS or stop-token handling.
    pub min_tokens: u32,
    /// Maximum number of reasoning ("thinking") tokens to emit before the
    /// reasoning section is force-closed. `None` means unlimited; the
    /// user-facing `-1` sentinel is normalized to `None` by the frontend before
    /// reaching this DTO, so only non-negative values are sent. Enforced
    /// engine-side (and only when a reasoning parser is configured).
    pub thinking_token_budget: Option<u64>,
    /// Number of log probabilities to return per generated token.
    ///
    /// `None` disables sample logprobs. `-1` requests the full vocabulary.
    pub logprobs: Option<i32>,
    /// Number of log probabilities to return per prompt token.
    ///
    /// `None` disables prompt logprobs. `-1` requests the full vocabulary.
    pub prompt_logprobs: Option<i32>,
    /// Minimum probability threshold for token sampling.
    pub min_p: f32,
    /// Frequency penalty applied by the sampler.
    pub frequency_penalty: f32,
    /// Presence penalty applied by the sampler.
    pub presence_penalty: f32,
    /// Repetition penalty applied by the sampler.
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    /// Parameters for detecting repetitive N-gram patterns. `None` disables
    /// detection.
    pub repetition_detection: Option<RepetitionDetectionParams>,
    /// Token IDs that stop generation.
    pub stop_token_ids: Vec<u32>,
    /// Primary EOS token ID used by engine-core's dedicated EOS stop path.
    ///
    /// This mirrors Python's internal `_eos_token_id` field and is derived by
    /// the frontend from tokenizer/model metadata rather than supplied directly
    /// by end users.
    #[serde(rename = "_eos_token_id")]
    pub eos_token_id: Option<u32>,
    /// Complete stop-token set used by engine-core for `min_tokens` masking.
    ///
    /// This mirrors Python's internal `_all_stop_token_ids` field and should
    /// contain explicit `stop_token_ids` plus any frontend-derived EOS token
    /// IDs.
    #[serde(rename = "_all_stop_token_ids")]
    pub all_stop_token_ids: BTreeSet<u32>,
    /// Logit biases to apply during sampling.
    /// Keys are token IDs
    pub logit_bias: Option<HashMap<u32, f32>>,
    /// Restrict output to these token IDs only.
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Tokenized bad words to avoid during generation.
    #[serde(rename = "_bad_words_token_ids")]
    pub bad_words_token_ids: Option<Vec<Vec<u32>>>,
    /// Parameters for configuring structured outputs (guided decoding).
    pub structured_outputs: Option<StructuredOutputsParams>,
    /// Specific token IDs for which log probabilities should be returned at
    /// each position.
    ///
    /// When set, the engine returns logprobs for exactly these tokens in
    /// addition to the sampled/scored token. Mutually exclusive with the
    /// `logprobs` count field in practice.
    pub logprob_token_ids: Option<Vec<u32>>,
    /// If `Some(true)`, the request will not attempt to read from the prefix
    /// cache; newly computed blocks may still populate the cache. `None`
    /// defers to engine-core defaults.
    pub skip_reading_prefix_cache: Option<bool>,
    /// Additional request parameters for custom extensions (from `vllm_xargs`).
    pub extra_args: Option<HashMap<String, serde_json::Value>>,
    /// Base steering vectors applied to both prefill and decode phases, in the
    /// inline form engine-core resolves. `None` means no steering.
    #[serde(default)]
    pub steering_vectors: Option<SteeringVectorSpec>,
    /// Phase-specific steering vectors added to the base during prefill only.
    #[serde(default)]
    pub prefill_steering_vectors: Option<SteeringVectorSpec>,
    /// Phase-specific steering vectors added to the base during decode only.
    #[serde(default)]
    pub decode_steering_vectors: Option<SteeringVectorSpec>,
    /// Reference to a pre-registered named steering module as `(name, scale)`.
    /// The worker resolves the named module and merges any inline overrides.
    #[serde(default)]
    pub steering_module_ref: Option<(String, f32)>,
    /// Per-request opt-in for activation-capture consumers, keyed by consumer
    /// name. Forwarded verbatim; engine-core's input processor resolves the raw
    /// spec into prefix-cache flags (offline admission).
    #[serde(default)]
    pub capture: Option<serde_json::Value>,
    /// Per-request activation-patching spec: a list of site entries. Forwarded
    /// verbatim; engine-core's input processor resolves the raw spec into
    /// prefix-cache flags (offline admission).
    #[serde(default)]
    pub patch: Option<serde_json::Value>,
    /// Request-level packed table of client-provided patch vectors, referenced
    /// by a patch entry's `source_inline` / mask `inline` row index. Forwarded
    /// verbatim (base64 binary wire form) like `patch`; omit-when-None keeps the
    /// wire payload compatible with clients that never set it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch_vectors: Option<serde_json::Value>,
    /// Per-request steering clamps applied to both prefill and decode
    /// phases, in the canonical [`SteeringClamps`] form Python's strict
    /// decoder expects (the HTTP/gRPC layers pack client input into it).
    /// Omit-when-None keeps the wire payload compatible with clients that
    /// never set it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steering_clamps: Option<SteeringClamps>,
    /// Phase-specific steering clamps applied during prefill only. Same
    /// canonical form as `steering_clamps`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_steering_clamps: Option<SteeringClamps>,
    /// Phase-specific steering clamps applied during decode only. Same
    /// canonical form as `steering_clamps`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_steering_clamps: Option<SteeringClamps>,
}

impl EngineCoreSamplingParams {
    /// Constructs a default sampling params for testing purposes only.
    pub fn for_test() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
            max_tokens: 65536,
            min_tokens: 0,
            thinking_token_budget: None,
            logprobs: None,
            prompt_logprobs: None,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            repetition_detection: None,
            stop_token_ids: Vec::new(),
            eos_token_id: None,
            all_stop_token_ids: BTreeSet::new(),
            logit_bias: None,
            allowed_token_ids: None,
            bad_words_token_ids: None,
            structured_outputs: None,
            logprob_token_ids: None,
            skip_reading_prefix_cache: None,
            extra_args: None,
            steering_vectors: None,
            prefill_steering_vectors: None,
            decode_steering_vectors: None,
            steering_module_ref: None,
            capture: None,
            patch: None,
            patch_vectors: None,
            steering_clamps: None,
            prefill_steering_clamps: None,
            decode_steering_clamps: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rmpv::Value;

    use super::EngineCoreSamplingParams;
    use crate::protocol::request::EngineCoreRequest;
    use crate::protocol::{
        ClampHookTable, SteeringClamps, SteeringLayerEntry, decode_msgpack, decode_value,
        encode_msgpack,
    };

    /// A real `sampling_params` is a sparse `omit_defaults` map; absent fields
    /// must fall back to defaults. `python_compat` can't catch this since Rust
    /// encodes full maps (see `engine_core_request_serializes_as_full_array`).
    #[test]
    fn decodes_sampling_params_with_omitted_defaults() {
        let sampling_params = Value::Map(vec![
            (
                Value::from("stop_token_ids"),
                Value::Array(vec![Value::from(151643u32)]),
            ),
            (Value::from("skip_reading_prefix_cache"), Value::from(false)),
        ]);
        let request = Value::Array(vec![
            Value::from("req-omit-defaults"),
            Value::Array(vec![
                Value::from(1u32),
                Value::from(2u32),
                Value::from(3u32),
            ]),
            Value::Nil,
            sampling_params,
            Value::Nil,
            Value::from(1.0f64),
        ]);

        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &request).unwrap();

        let decoded: EngineCoreRequest = decode_msgpack(&bytes)
            .expect("a real omit_defaults request must decode (regression: missing field)");

        assert_eq!(decoded.request_id, "req-omit-defaults");
        let sampling = decoded.sampling_params.expect("sampling params present");

        assert_eq!(sampling.stop_token_ids, vec![151643]);
        assert_eq!(sampling.skip_reading_prefix_cache, Some(false));

        // Omitted fields -> Python defaults.
        assert_eq!(sampling.temperature, 1.0);
        assert_eq!(sampling.top_p, 1.0);
        assert_eq!(sampling.top_k, 0);
        assert_eq!(sampling.seed, None);
        assert_eq!(sampling.max_tokens, 16);
        assert_eq!(sampling.min_tokens, 0);
        assert_eq!(sampling.min_p, 0.0);
        assert_eq!(sampling.frequency_penalty, 0.0);
        assert_eq!(sampling.presence_penalty, 0.0);
        assert_eq!(sampling.repetition_penalty, 1.0);
        assert_eq!(sampling.repetition_detection, None);
        assert_eq!(sampling.logprobs, None);
        assert_eq!(sampling.prompt_logprobs, None);
        assert_eq!(sampling.eos_token_id, None);
        assert!(sampling.all_stop_token_ids.is_empty());
    }

    #[test]
    fn steering_and_capture_use_python_field_names_and_int_layer_keys() {
        let mut hook = HashMap::new();
        hook.insert(
            7u32,
            SteeringLayerEntry {
                vector: vec![1.0, 2.0],
                scale: 0.5,
            },
        );
        let spec = HashMap::from([("pre_attn".to_string(), hook)]);

        let params = EngineCoreSamplingParams {
            steering_vectors: Some(spec),
            steering_module_ref: Some(("creativity".to_string(), 1.0)),
            capture: Some(serde_json::json!({
                "filesystem": {"tag": "t", "positions": "last_prompt"}
            })),
            ..EngineCoreSamplingParams::for_test()
        };

        // The sampling params serialize as a field-name map (`to_vec_named`).
        let value = decode_value(&encode_msgpack(&params).unwrap()).unwrap();
        let map = match value {
            Value::Map(map) => map,
            other => panic!("expected map, got {other:?}"),
        };
        let get = |key: &str| map.iter().find(|(k, _)| k.as_str() == Some(key)).map(|(_, v)| v);

        // `steering_module_ref` is a 2-element `(name, scale)` array.
        match get("steering_module_ref").expect("steering_module_ref present") {
            Value::Array(a) => {
                assert_eq!(a.len(), 2);
                assert_eq!(a[0].as_str(), Some("creativity"));
            }
            other => panic!("expected array, got {other:?}"),
        }

        // `capture` is forwarded verbatim as a map.
        assert!(matches!(get("capture"), Some(Value::Map(_))));

        // `steering_vectors` is {hook: {layer_idx: entry}}; the layer key MUST be
        // a msgpack integer so Python decodes it into `dict[int, ...]`.
        let hooks = match get("steering_vectors").expect("steering_vectors present") {
            Value::Map(map) => map,
            other => panic!("expected map, got {other:?}"),
        };
        let (hook_name, layers) = &hooks[0];
        assert_eq!(hook_name.as_str(), Some("pre_attn"));
        let layers = match layers {
            Value::Map(map) => map,
            other => panic!("expected map, got {other:?}"),
        };
        let (layer_key, _) = &layers[0];
        assert!(
            layer_key.is_i64() || layer_key.is_u64(),
            "layer key must be an integer, got {layer_key:?}"
        );
        assert_eq!(layer_key.as_u64(), Some(7));
    }

    #[test]
    fn patch_spec_is_forwarded_verbatim_as_a_list() {
        let spec = serde_json::json!([
            {
                "layer": 14,
                "hook": "post_block",
                "dest_position": 6,
                "source_run": "clean",
                "source_position": 6,
                "alpha": 1.0,
            }
        ]);
        let params = EngineCoreSamplingParams {
            patch: Some(spec.clone()),
            ..EngineCoreSamplingParams::for_test()
        };

        // `patch` serializes as a msgpack array under the `patch` field name,
        // and round-trips back to the same JSON list engine-core admits.
        let bytes = encode_msgpack(&params).unwrap();
        let value = decode_value(&bytes).unwrap();
        let map = match value {
            Value::Map(map) => map,
            other => panic!("expected map, got {other:?}"),
        };
        let get = |key: &str| map.iter().find(|(k, _)| k.as_str() == Some(key)).map(|(_, v)| v);
        assert!(
            matches!(get("patch"), Some(Value::Array(_))),
            "patch must forward as a msgpack array"
        );

        let decoded: EngineCoreSamplingParams = decode_msgpack(&bytes).unwrap();
        assert_eq!(decoded.patch, Some(spec));
    }

    #[test]
    fn patch_defaults_to_none_when_absent() {
        // A request without a patch spec leaves the field `None`, so a missing
        // `patch` key in the wire payload decodes cleanly (serde default).
        let params = EngineCoreSamplingParams::for_test();
        assert!(params.patch.is_none());
        let decoded: EngineCoreSamplingParams =
            decode_msgpack(&encode_msgpack(&params).unwrap()).unwrap();
        assert!(decoded.patch.is_none());
    }

    #[test]
    fn patch_vectors_forwarded_verbatim_and_omitted_when_none() {
        // The packed table forwards verbatim under `patch_vectors` (like
        // `patch`), and its key is omitted from the wire payload when None.
        let table = serde_json::json!({
            "dtype": "float32",
            "shape": [2, 4],
            "data": "AAAAAA==",
        });
        let params = EngineCoreSamplingParams {
            patch_vectors: Some(table.clone()),
            ..EngineCoreSamplingParams::for_test()
        };
        let bytes = encode_msgpack(&params).unwrap();
        let decoded: EngineCoreSamplingParams = decode_msgpack(&bytes).unwrap();
        assert_eq!(decoded.patch_vectors, Some(table));

        // Absent by default: the key is skipped, decodes cleanly to None.
        let plain = EngineCoreSamplingParams::for_test();
        assert!(plain.patch_vectors.is_none());
        let decoded: EngineCoreSamplingParams =
            decode_msgpack(&encode_msgpack(&plain).unwrap()).unwrap();
        assert!(decoded.patch_vectors.is_none());
    }

    #[test]
    fn steering_clamps_use_python_field_names_and_are_omitted_when_none() {
        // The three clamp fields ride under their exact Python msgspec field
        // names in the canonical `{"hooks": {...}}` form, and are skipped on
        // the wire when None.
        let table = ClampHookTable {
            shape: vec![1, 2],
            layer_indices: vec![5],
            data: [1.0f64, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
            lo: vec![-2.0],
            hi: vec![2.0],
            strength: vec![1.0],
        };
        let clamps = SteeringClamps {
            hooks: HashMap::from([("post_attn".to_owned(), table.clone())]),
        };
        let params = EngineCoreSamplingParams {
            steering_clamps: Some(clamps.clone()),
            prefill_steering_clamps: Some(clamps.clone()),
            decode_steering_clamps: Some(clamps.clone()),
            ..EngineCoreSamplingParams::for_test()
        };

        let bytes = encode_msgpack(&params).unwrap();
        let value = decode_value(&bytes).unwrap();
        let map = match value {
            Value::Map(map) => map,
            other => panic!("expected map, got {other:?}"),
        };
        let get = |key: &str| map.iter().find(|(k, _)| k.as_str() == Some(key)).map(|(_, v)| v);
        for key in [
            "steering_clamps",
            "prefill_steering_clamps",
            "decode_steering_clamps",
        ] {
            let Some(Value::Map(outer)) = get(key) else {
                panic!("{key} must ride as a msgpack map under its exact wire name");
            };
            assert!(
                outer.iter().any(|(k, _)| k.as_str() == Some("hooks")),
                "{key} must carry the canonical hooks nesting"
            );
        }

        let decoded: EngineCoreSamplingParams = decode_msgpack(&bytes).unwrap();
        assert_eq!(decoded.steering_clamps, Some(clamps.clone()));
        assert_eq!(decoded.prefill_steering_clamps, Some(clamps.clone()));
        assert_eq!(decoded.decode_steering_clamps, Some(clamps));

        // Absent by default: all three keys are skipped on the wire and decode
        // cleanly to None.
        let plain = EngineCoreSamplingParams::for_test();
        let plain_bytes = encode_msgpack(&plain).unwrap();
        let plain_map = match decode_value(&plain_bytes).unwrap() {
            Value::Map(map) => map,
            other => panic!("expected map, got {other:?}"),
        };
        for key in [
            "steering_clamps",
            "prefill_steering_clamps",
            "decode_steering_clamps",
        ] {
            assert!(
                !plain_map.iter().any(|(k, _)| k.as_str() == Some(key)),
                "{key} must be omitted from the wire payload when None"
            );
        }
        let decoded: EngineCoreSamplingParams = decode_msgpack(&plain_bytes).unwrap();
        assert!(decoded.steering_clamps.is_none());
        assert!(decoded.prefill_steering_clamps.is_none());
        assert!(decoded.decode_steering_clamps.is_none());
    }

    #[test]
    fn clamp_row_data_rides_msgpack_bin_with_true_infinities() {
        // Python msgspec strictly expects `bytes` (msgpack bin) for the row
        // data and native non-finite floats for open bounds — an integer
        // array or a null sentinel would fail the engine-side decode.
        let rows: Vec<u8> = [1.5f64, -2.5].iter().flat_map(|v| v.to_le_bytes()).collect();
        let table = ClampHookTable {
            shape: vec![1, 2],
            layer_indices: vec![14],
            data: rows.clone(),
            lo: vec![f64::NEG_INFINITY],
            hi: vec![4.0],
            strength: vec![0.5],
        };
        let clamps = SteeringClamps {
            hooks: HashMap::from([("post_block".to_owned(), table)]),
        };
        let params = EngineCoreSamplingParams {
            steering_clamps: Some(clamps.clone()),
            ..EngineCoreSamplingParams::for_test()
        };

        let bytes = encode_msgpack(&params).unwrap();
        let value = decode_value(&bytes).unwrap();
        let Value::Map(map) = value else {
            panic!("expected map");
        };
        let clamps_value = map
            .iter()
            .find(|(k, _)| k.as_str() == Some("steering_clamps"))
            .map(|(_, v)| v)
            .expect("steering_clamps present");
        let Value::Map(outer) = clamps_value else {
            panic!("expected hooks map");
        };
        let Some((_, Value::Map(hooks))) = outer.iter().find(|(k, _)| k.as_str() == Some("hooks"))
        else {
            panic!("expected hooks key");
        };
        let Some((_, Value::Map(blob))) =
            hooks.iter().find(|(k, _)| k.as_str() == Some("post_block"))
        else {
            panic!("expected post_block table");
        };
        let data = blob.iter().find(|(k, _)| k.as_str() == Some("data")).map(|(_, v)| v);
        assert!(
            matches!(data, Some(Value::Binary(b)) if *b == rows),
            "row data must be a msgpack bin of the raw f64 bytes, got {data:?}"
        );
        let lo = blob.iter().find(|(k, _)| k.as_str() == Some("lo")).map(|(_, v)| v);
        assert!(
            matches!(lo, Some(Value::Array(vals))
                if matches!(vals[0], Value::F64(f) if f == f64::NEG_INFINITY)),
            "open bounds must ride as native -inf, got {lo:?}"
        );

        let decoded: EngineCoreSamplingParams = decode_msgpack(&bytes).unwrap();
        assert_eq!(decoded.steering_clamps, Some(clamps));
    }

    #[test]
    fn clamp_data_serializes_as_base64_in_json() {
        // The module-broadcast kwargs are built via serde_json before the
        // rmpv utility encoding, so the JSON form must carry base64 (which
        // Python's from_obj wire-map branch decodes) — never an int array.
        let rows: Vec<u8> = [3.0f64].iter().flat_map(|v| v.to_le_bytes()).collect();
        let clamps = SteeringClamps {
            hooks: HashMap::from([(
                "post_attn".to_owned(),
                ClampHookTable {
                    shape: vec![1, 1],
                    layer_indices: vec![2],
                    data: rows,
                    lo: vec![0.0],
                    hi: vec![1.0],
                    strength: vec![1.0],
                },
            )]),
        };
        let json = serde_json::to_value(&clamps).unwrap();
        let data = &json["hooks"]["post_attn"]["data"];
        assert!(data.is_string(), "JSON data must be base64, got {data:?}");
        let decoded: SteeringClamps = serde_json::from_value(json).unwrap();
        assert_eq!(decoded, clamps);
    }
}
