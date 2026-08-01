//! Shared helpers for surfacing activation-capture results on API responses.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;
use serde_json::{Map, Value};
use vllm_engine_core_client::protocol::CaptureResult;

/// Per-consumer capture result surfaced on the OpenAI response body.
///
/// Mirror of Python's `CaptureResultResponse`: the lifecycle `status`, an
/// optional first `error`, and a consumer-specific `payload` object.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CaptureResultResponse {
    /// Lifecycle state: `pending` | `ok` | `partial_error` | `error` |
    /// `not_requested`.
    pub status: String,
    /// First error message, if any.
    pub error: Option<String>,
    /// Consumer-specific payload, always a JSON object (the filesystem consumer
    /// emits `{"paths": [...]}`).
    pub payload: Value,
}

/// Convert engine-core capture results into the response dict, or `None` when
/// empty so serializers omit the field for the common uncaptured request.
///
/// Mirrors Python's `_build_capture_results_response`.
pub fn build_capture_results_response(
    results: &HashMap<String, CaptureResult>,
) -> Option<BTreeMap<String, CaptureResultResponse>> {
    if results.is_empty() {
        return None;
    }
    Some(
        results
            .iter()
            .map(|(name, result)| {
                (
                    name.clone(),
                    CaptureResultResponse {
                        status: result.status.clone(),
                        error: result.error.clone(),
                        payload: coerce_payload(result.payload.as_ref()),
                    },
                )
            })
            .collect(),
    )
}

/// Coerce an opaque consumer payload into a JSON object, mirroring Python's
/// `_capture_result_to_response_payload`: objects pass through, `null` becomes
/// `{}`, arrays become `{"items": [...]}`, and scalars become `{"value": ...}`.
fn coerce_payload(payload: Option<&Value>) -> Value {
    match payload {
        None | Some(Value::Null) => Value::Object(Map::new()),
        Some(Value::Object(_)) => payload.cloned().unwrap_or_else(|| Value::Object(Map::new())),
        Some(Value::Array(items)) => {
            let mut map = Map::new();
            map.insert("items".to_string(), Value::Array(items.clone()));
            Value::Object(map)
        }
        Some(other) => {
            let mut map = Map::new();
            map.insert("value".to_string(), other.clone());
            Value::Object(map)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(status: &str, payload: Option<Value>) -> CaptureResult {
        CaptureResult {
            key: None,
            status: status.to_string(),
            error: None,
            payload,
        }
    }

    #[test]
    fn empty_results_yield_none() {
        assert!(build_capture_results_response(&HashMap::new()).is_none());
    }

    #[test]
    fn coerces_payload_shapes() {
        let results = HashMap::from([
            (
                "fs".to_string(),
                result("ok", Some(serde_json::json!({ "paths": ["/a.bin"] }))),
            ),
            (
                "list".to_string(),
                result("ok", Some(serde_json::json!([1, 2]))),
            ),
            ("none".to_string(), result("ok", None)),
            (
                "scalar".to_string(),
                result("ok", Some(serde_json::json!(7))),
            ),
        ]);
        let out = build_capture_results_response(&results).expect("non-empty");
        assert_eq!(
            out["fs"].payload,
            serde_json::json!({ "paths": ["/a.bin"] })
        );
        assert_eq!(out["list"].payload, serde_json::json!({ "items": [1, 2] }));
        assert_eq!(out["none"].payload, serde_json::json!({}));
        assert_eq!(out["scalar"].payload, serde_json::json!({ "value": 7 }));
    }
}
