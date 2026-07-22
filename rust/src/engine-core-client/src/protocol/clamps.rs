//! Canonical directional-clamp spec, mirroring Python's `SteeringClamps` /
//! `ClampHookTable` msgspec Structs (`vllm/config/steering_types.py`).
//!
//! This is the ONLY clamp shape that crosses the engine-core wire: Python's
//! `SamplingParams.steering_clamps` fields are strictly typed as
//! `SteeringClamps | None` and its decoder rejects anything else (including
//! the legacy verbatim `{hook: {layer: [entries]}}` JSON this frontend used
//! to forward). The northbound HTTP/gRPC layers accept the human-authorable
//! entry-list and base64-packed input shapes and pack them into this struct
//! before lowering — see `vllm-server`'s `routes::openai::utils::clamps`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Format-aware serde for the packed row bytes.
///
/// - Binary formats (msgpack via rmp-serde — the engine-core wire and the
///   collective_rpc hop) carry the rows as a native `bin`, matching what
///   Python msgspec emits/expects for a `bytes` field.
/// - Human-readable formats (serde_json — the module-broadcast kwargs are
///   built as JSON `Value`s before the rmpv utility encoding, and module
///   files on disk) carry them base64-encoded, which Python's
///   `SteeringClamps.from_obj` wire-map branch decodes.
mod data_bytes {
    use base64::Engine as _;
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(data: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            serializer.serialize_str(&base64::engine::general_purpose::STANDARD.encode(data))
        } else {
            serializer.serialize_bytes(data)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        if deserializer.is_human_readable() {
            let encoded = String::deserialize(deserializer)?;
            base64::engine::general_purpose::STANDARD
                .decode(encoded.as_bytes())
                .map_err(|err| D::Error::custom(format!("invalid base64 clamp data: {err}")))
        } else {
            serde_bytes::ByteBuf::deserialize(deserializer).map(serde_bytes::ByteBuf::into_vec)
        }
    }
}

/// One hook point's clamp rows in the canonical packed layout.
///
/// `data` holds `shape[0]` float64 little-endian rows of width `shape[1]`,
/// stored exactly as submitted (NOT unit-normalized — the engine normalizes
/// at its consumption boundaries, so bound semantics stay in unit-projection
/// space regardless of the client's vector scale). Row order within a layer
/// is semantic: it is the tier-merge concat order and the per-site K budget.
///
/// `lo`/`hi` carry true `±inf` for open bounds — msgpack encodes non-finite
/// floats natively, so no `null` sentinel exists at this layer.
///
/// The `data` field is format-aware (see [`data_bytes`]): msgpack `bin` on
/// binary wires, base64 in JSON — a plain `Vec<u8>` would serialize as an
/// integer array and fail the strict engine-side decode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClampHookTable {
    /// Row-matrix shape as `[num_rows, hidden_size]`.
    pub shape: Vec<u32>,
    /// Layer index for each row, in row order (repeats allowed). Length must
    /// equal `shape[0]`.
    pub layer_indices: Vec<u32>,
    /// `num_rows * hidden_size` float64 little-endian direction values,
    /// row-major.
    #[serde(with = "data_bytes")]
    pub data: Vec<u8>,
    /// Per-row lower bound in unit-projection space (`-inf` = open below).
    pub lo: Vec<f64>,
    /// Per-row upper bound (`+inf` = open above).
    pub hi: Vec<f64>,
    /// Per-row blend strength in `[0, 1]`.
    pub strength: Vec<f64>,
}

/// The canonical clamp tier: hook-point name → packed row table.
///
/// Wire form is a single-key map `{"hooks": {...}}` — the extra nesting level
/// is what lets Python's decoder distinguish this struct from the legacy
/// hook-keyed dicts (`"hooks"` is not a valid hook-point name, and unknown
/// fields are forbidden on the Python side).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SteeringClamps {
    /// Hook-point name → clamp row table.
    #[serde(default)]
    pub hooks: HashMap<String, ClampHookTable>,
}

impl SteeringClamps {
    /// `true` when no hook carries any rows.
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }
}
