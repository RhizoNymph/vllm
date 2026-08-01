//! Per-request activation-capture result types returned by engine-core.
//!
//! Mirrors Python's `vllm.v1.capture.types.CaptureResult` (a `@dataclass`,
//! encoded by msgspec as a msgpack map keyed by field name) carried on
//! [`crate::protocol::EngineCoreOutput::capture_results`].

use serde::{Deserialize, Serialize};

/// Terminal per-consumer capture result.
///
/// Attached, keyed by consumer name, to the engine-core output of a request
/// that opted into activation capture. The `payload` is consumer-specific and
/// opaque to the frontend (the filesystem consumer emits `{"paths": [...]}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureResult {
    /// Opaque capture key — Python's `(request_id, layer, hook)` tuple.
    /// Preserved for round-tripping but not interpreted by the frontend.
    #[serde(default)]
    pub key: Option<serde_json::Value>,
    /// Lifecycle state: `pending` | `ok` | `partial_error` | `error` |
    /// `not_requested`.
    pub status: String,
    /// First error message, if any.
    #[serde(default)]
    pub error: Option<String>,
    /// Consumer-specific opaque payload.
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
}
