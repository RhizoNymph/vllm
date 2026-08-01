//! Per-request activation-steering types in the inline form accepted by
//! engine-core.
//!
//! The northbound HTTP/gRPC APIs accept steering vectors in a compact packed
//! wire format (base64 bytes + dtype/shape metadata). The frontend decodes that
//! into the *inline* form modeled here before forwarding it southbound on
//! [`crate::protocol::EngineCoreSamplingParams`]. Engine-core / the worker then
//! resolve these entries into model-dtype arrays — see Python's
//! `vllm.config.steering_types` (`SteeringVectorSpec`, `SteeringLayerEntry`).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// One steering entry for a single `(hook, layer)` position: a flat vector plus
/// a scalar multiplier.
///
/// Serializes to the dict form of Python's `SteeringLayerEntry`
/// (`{"vector": [...], "scale": ...}`), which the engine-core/worker accepts
/// directly via `_split_entry` and scales into model-dtype arrays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SteeringLayerEntry {
    /// Flat steering vector with one element per hidden dimension.
    pub vector: Vec<f32>,
    /// Scalar multiplier applied to `vector` before it is added to the residual.
    pub scale: f32,
}

/// Per-request steering specification in the inline form accepted by
/// engine-core: hook-point name → layer index → entry.
///
/// The layer-index map uses an integer key (`u32`) so it serializes to msgpack
/// integer keys and Python decodes it into `dict[int, SteeringLayerEntry]`
/// rather than `dict[str, ...]`.
pub type SteeringVectorSpec = HashMap<String, HashMap<u32, SteeringLayerEntry>>;
