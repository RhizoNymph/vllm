//! Per-request SAE steering (feature-surgery) types in the typed form
//! accepted by engine-core.
//!
//! These mirror Python's `vllm.config.sae_steering_types` field-for-field
//! (`SAEClampEntry`, `SAEClampSpec`, `SAEFullReconstructionSpec`). The
//! frontend forwards them southbound on
//! [`crate::protocol::EngineCoreSamplingParams`]; engine-core / the worker
//! resolve the referenced named SAE modules and apply the clamps.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

/// Discriminator for the two clamp variants, mirroring Python's
/// `SAEClampKind` literal (`"absolute" | "additive"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SaeClampKind {
    /// Set the feature activation to `value` (`f_i := value`).
    Absolute,
    /// Shift the feature activation by `value` (`f_i := f_i + value`).
    Additive,
}

/// Phase tier for an SAE spec, mirroring Python's `SAEClampPhase` literal
/// (`"both" | "prefill" | "decode"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SaePhase {
    /// Apply during both prefill and decode (the default).
    #[default]
    Both,
    /// Apply during prefill only.
    Prefill,
    /// Apply during decode only.
    Decode,
}

/// One feature clamp inside an [`SaeClampSpec`] /
/// [`SaeFullReconstructionSpec`], mirroring Python's `SAEClampEntry`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SaeClampEntry {
    /// Index of the SAE feature to clamp (column of `W_dec`).
    pub feature_idx: u32,
    /// Clamp variant: absolute set or additive shift.
    pub kind: SaeClampKind,
    /// Target value (absolute) or offset (additive).
    pub value: f64,
    /// For additive clamps: only apply when the live activation is positive.
    #[serde(default)]
    pub only_if_active: bool,
}

/// `hook_point → layer_idx → clamp entries` for a single module.
///
/// The layer map uses an integer key (`u32`) so it serializes to msgpack
/// integer keys: Python engine-core decodes `SamplingParams` with a strict
/// `msgspec.msgpack.Decoder` whose `clamps` field is `dict[int, ...]`, which
/// rejects string layer keys outright. (`serde_json::Value` cannot express
/// int map keys, hence the typed mirror.)
pub type SaeClampHookMap = HashMap<String, BTreeMap<u32, Vec<SaeClampEntry>>>;

/// Per-request clamp directive for one SAE module (delta intervention),
/// mirroring Python's `SAEClampSpec`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SaeClampSpec {
    /// Name of the pre-registered SAE module (kind `sae_delta`).
    pub module_name: String,
    /// Which feature activations to clamp on which `(hook, layer)` sites.
    /// Must be non-empty; engine-core validates semantics at admission.
    #[serde(deserialize_with = "deserialize_clamp_hook_map")]
    pub clamps: SaeClampHookMap,
    /// Phase tier the clamps apply to.
    #[serde(default)]
    pub phase: SaePhase,
}

/// Per-request directive for the SAE full-reconstruction path (residual
/// replacement), mirroring Python's `SAEFullReconstructionSpec`. Unlike
/// [`SaeClampSpec`], `clamps` may be empty (pure reconstruction).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SaeFullReconstructionSpec {
    /// Name of the pre-registered SAE module (kind `sae_full_reconstruction`).
    pub module_name: String,
    /// Optional clamps applied to the SAE activations before the decoder
    /// pass. Empty means bare reconstruction.
    #[serde(default, deserialize_with = "deserialize_clamp_hook_map")]
    pub clamps: SaeClampHookMap,
    /// Phase tier the reconstruction applies to.
    #[serde(default)]
    pub phase: SaePhase,
}

/// Layer-map key that tolerates every inbound key encoding while always
/// serializing (via the plain `u32` in [`SaeClampHookMap`]) to a msgpack
/// integer:
///
/// * msgpack integer keys (southbound round-trips),
/// * JSON string keys like `"20"` (JSON objects cannot have int keys),
/// * `serde` `Content`-buffered string keys (HTTP request structs with a
///   `#[serde(flatten)]` catch-all buffer all fields, and the buffered map-key
///   deserializer does not coerce strings to integers on its own).
struct FlexLayerKey(u32);

impl<'de> Deserialize<'de> for FlexLayerKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct KeyVisitor;

        impl serde::de::Visitor<'_> for KeyVisitor {
            type Value = FlexLayerKey;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a non-negative integer layer index (int or int-valued string)")
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                u32::try_from(v).map(FlexLayerKey).map_err(E::custom)
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                u32::try_from(v).map(FlexLayerKey).map_err(E::custom)
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                v.parse::<u32>()
                    .map(FlexLayerKey)
                    .map_err(|_| E::custom(format!("invalid layer key `{v}`: expected an integer")))
            }
        }

        deserializer.deserialize_any(KeyVisitor)
    }
}

/// Deserialize an [`SaeClampHookMap`] accepting flexible layer-key encodings
/// (see [`FlexLayerKey`]).
fn deserialize_clamp_hook_map<'de, D>(deserializer: D) -> Result<SaeClampHookMap, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(HashMap::<String, LayerSeq>::deserialize(deserializer)?
        .into_iter()
        .map(|(hook, LayerSeq(layers))| {
            (
                hook,
                layers.into_iter().map(|(key, entries)| (key.0, entries)).collect(),
            )
        })
        .collect())
}

/// Helper that deserializes one hook's layer map as a key/value sequence so
/// [`FlexLayerKey`] does not need `Ord`/`Hash` implementations.
struct LayerSeq(Vec<(FlexLayerKey, Vec<SaeClampEntry>)>);

impl<'de> Deserialize<'de> for LayerSeq {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct MapVisitor;

        impl<'de> serde::de::Visitor<'de> for MapVisitor {
            type Value = LayerSeq;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a map of layer index to clamp-entry list")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut layers = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some(entry) = map.next_entry::<FlexLayerKey, Vec<SaeClampEntry>>()? {
                    layers.push(entry);
                }
                Ok(LayerSeq(layers))
            }
        }

        deserializer.deserialize_map(MapVisitor)
    }
}
