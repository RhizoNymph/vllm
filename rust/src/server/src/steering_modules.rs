//! Startup loading and engine broadcast of named steering modules.
//!
//! Named modules are loaded from JSON files at server startup and broadcast to
//! every engine worker via `collective_rpc`, so requests can reference them by
//! `steering_name` without re-sending the (potentially large) vector blobs on
//! every request. This mirrors the Python API server's startup path
//! (`SteeringModuleRegistry` + `register_steering_modules` /
//! `pre_materialize_steering_module`).

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::Context as _;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::info;
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::{SteeringLayerEntry, SteeringVectorSpec};

use crate::config::SteeringModulePath;
use crate::routes::openai::utils::steering::{
    SteeringDecodeError, SteeringSpecPacked, steering_dtype_element_size, unpack_steering_spec,
};

/// Kind discriminator for a named steering module, mirroring Python's
/// `SteeringModuleKind` literal values. Legacy payloads without a `kind`
/// field default to `additive`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SteeringModuleKind {
    /// Additive steering vectors (`vectors` / `prefill_vectors` /
    /// `decode_vectors` tiers).
    #[serde(rename = "additive")]
    Additive,
    /// SAE feature-surgery delta intervention (`sae_manifest` + weights).
    #[serde(rename = "sae_delta")]
    SaeDelta,
    /// SAE full residual reconstruction (`sae_manifest` + weights, incl.
    /// `decoder_bias`).
    #[serde(rename = "sae_full_reconstruction")]
    SaeFullReconstruction,
}

impl SteeringModuleKind {
    /// Wire string for error messages (matches the serde rename).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Additive => "additive",
            Self::SaeDelta => "sae_delta",
            Self::SaeFullReconstruction => "sae_full_reconstruction",
        }
    }

    /// Whether this is one of the two SAE kinds (which carry a manifest and
    /// packed weights instead of additive vector tiers, and are exempt from
    /// pre-materialization — the worker attaches SAE buffers during the
    /// register RPC itself).
    pub fn is_sae(self) -> bool {
        !matches!(self, Self::Additive)
    }
}

/// JSON-safe SAE manifest, mirroring Python's `_sae_manifest_to_dict` /
/// `sae_manifest_from_dict` shape field-for-field. Semantic validation
/// (activation params, shape invariants, site ownership) stays on the worker;
/// this struct only pins the structural shape the worker's decoder requires.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaeManifest {
    /// Model residual width the SAE operates on.
    pub d_model: u64,
    /// SAE latent dimension.
    pub d_sae: u64,
    /// Activation function name (`relu` | `jumprelu` | `topk`).
    pub activation: String,
    /// `(layer_idx, hook_point)` sites the module covers. Serializes as
    /// `[[layer, hook], ...]`, the pair-list form `sae_manifest_from_dict`
    /// expects.
    pub layers: Vec<(u32, String)>,
    /// Feature indices that per-request clamps may target.
    pub clampable_features: Vec<u64>,
    /// Activation-specific parameters (e.g. `{"threshold": x}`).
    #[serde(default)]
    pub activation_params: serde_json::Map<String, Value>,
    /// Optional checkpoint path recorded for reload; unused by the Rust
    /// frontend (weights always travel inline as `sae_weights`).
    #[serde(default)]
    pub weights_uri: Option<String>,
}

/// One tensor in the wire-safe packed form: base64-encoded little-endian
/// bytes plus dtype/shape metadata. Torch tensors do not survive the
/// `collective_rpc` msgpack hop, so SAE weights cross as these dicts and the
/// worker's `_coerce_sae_weights_wire` rebuilds tensors on arrival.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedTensor {
    /// Numpy dtype string (`float32` | `float16` | `bfloat16` | `float64`).
    pub dtype: String,
    /// Tensor shape (any rank; element count is the product).
    pub shape: Vec<u64>,
    /// Base64-encoded contiguous little-endian tensor bytes.
    pub data: String,
}

/// SAE weights payload: `"layer:hook"` site key → tensor-name → packed
/// tensor. Tensor names pass through opaquely (typically `encoder_weight`,
/// `encoder_bias`, `decoder_weight`, plus `decoder_bias` for the
/// full-reconstruction kind).
pub type SaeWeights = HashMap<String, HashMap<String, PackedTensor>>;

/// Broadcast payload for one named module: a `kind` discriminator plus the
/// kind-specific fields in the form the worker resolves. Field names match
/// the keys read by the worker's `register_steering_modules` (Python
/// `dump_for_broadcast`).
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct SteeringModuleBroadcast {
    /// Module kind; SAE kinds skip pre-materialization.
    pub kind: SteeringModuleKind,
    /// Base vectors applied to both prefill and decode (additive only).
    pub vectors: Option<SteeringVectorSpec>,
    /// Prefill-only additions (additive only).
    pub prefill_vectors: Option<SteeringVectorSpec>,
    /// Decode-only additions (additive only).
    pub decode_vectors: Option<SteeringVectorSpec>,
    /// SAE manifest (SAE kinds only).
    pub sae_manifest: Option<SaeManifest>,
    /// Packed SAE weights keyed by `"layer:hook"` (SAE kinds only).
    pub sae_weights: Option<SaeWeights>,
}

/// Errors raised while loading a named steering module from disk.
#[derive(Debug, thiserror::Error)]
pub enum SteeringModuleLoadError {
    #[error("steering module file not found: {path}")]
    NotFound { path: String },
    #[error("failed to read steering module file {path}: {message}")]
    Read { path: String, message: String },
    #[error("steering module file {path} is not valid JSON: {message}")]
    Json { path: String, message: String },
    #[error("steering module file {path} must contain a JSON object")]
    NotObject { path: String },
    #[error("steering module tier `{tier}` is malformed: {message}")]
    Tier { tier: &'static str, message: String },
    #[error("steering module tier `{tier}`: {source}")]
    Decode {
        tier: &'static str,
        #[source]
        source: SteeringDecodeError,
    },
    #[error(
        "steering module has unknown kind {kind} (expected \"additive\", \"sae_delta\", or \
         \"sae_full_reconstruction\")"
    )]
    UnknownKind { kind: String },
    #[error("steering module kind `additive` must not carry `{field}`")]
    SaeFieldOnAdditive { field: &'static str },
    #[error("steering module kind `{kind}` requires an `sae_manifest` object")]
    MissingSaeManifest { kind: &'static str },
    #[error("steering module kind `{kind}` must not carry additive vector tier `{tier}`")]
    AdditiveTierOnSae {
        kind: &'static str,
        tier: &'static str,
    },
    #[error("steering module `sae_manifest` is malformed: {message}")]
    SaeManifest { message: String },
    #[error(
        "steering module kind `{kind}` requires inline `sae_weights` (without them the worker's \
         SAE buffers stay zero-filled and every clamp silently no-ops)"
    )]
    MissingSaeWeights { kind: &'static str },
    #[error("steering module `sae_weights` site `{site}`: {message}")]
    SaeWeights { site: String, message: String },
}

/// Load a single named steering module from its JSON file.
///
/// Additive modules (`kind` absent or `"additive"`): each of the `vectors`,
/// `prefill_vectors`, and `decode_vectors` tiers may be either the inline
/// shape (`{hook: {layer: [floats] | {vector, scale}}}`, layer keys as
/// strings or ints) or the packed wire shape (per-hook
/// `{dtype, shape, layer_indices, data, scales}`), detected per tier.
///
/// SAE modules (`kind` `"sae_delta"` / `"sae_full_reconstruction"`): the file
/// carries `sae_manifest` plus `sae_weights` keyed by `"layer:hook"`, each
/// site a map of tensor name → `{dtype, shape, data}` packed blob.
pub fn load_steering_module(
    path: &str,
) -> Result<SteeringModuleBroadcast, SteeringModuleLoadError> {
    let file_path = Path::new(path);
    if !file_path.exists() {
        return Err(SteeringModuleLoadError::NotFound {
            path: path.to_owned(),
        });
    }
    let contents =
        std::fs::read_to_string(file_path).map_err(|err| SteeringModuleLoadError::Read {
            path: path.to_owned(),
            message: format!("{err}"),
        })?;
    let data: Value =
        serde_json::from_str(&contents).map_err(|err| SteeringModuleLoadError::Json {
            path: path.to_owned(),
            message: format!("{err}"),
        })?;
    let Value::Object(obj) = data else {
        return Err(SteeringModuleLoadError::NotObject {
            path: path.to_owned(),
        });
    };

    parse_module(&obj)
}

/// Parse one module object into the broadcast form, branching on the `kind`
/// discriminator (default `additive`). Shared by the file loader and the
/// runtime register endpoint (which receives the same shape inline in the
/// request body).
pub fn parse_module(
    obj: &serde_json::Map<String, Value>,
) -> Result<SteeringModuleBroadcast, SteeringModuleLoadError> {
    let kind = match obj.get("kind") {
        None | Some(Value::Null) => SteeringModuleKind::Additive,
        Some(value) => serde_json::from_value(value.clone()).map_err(|_| {
            SteeringModuleLoadError::UnknownKind {
                kind: value.to_string(),
            }
        })?,
    };
    if !kind.is_sae() {
        return parse_additive_module(obj);
    }

    // SAE kinds: additive tier keys are rejected outright (the registries are
    // disjoint) and both the manifest and inline weights are mandatory.
    for tier in ["vectors", "prefill_vectors", "decode_vectors"] {
        if obj.get(tier).is_some_and(|v| !v.is_null()) {
            return Err(SteeringModuleLoadError::AdditiveTierOnSae {
                kind: kind.as_str(),
                tier,
            });
        }
    }
    let manifest_value = match obj.get("sae_manifest") {
        Some(value) if !value.is_null() => value,
        _ => {
            return Err(SteeringModuleLoadError::MissingSaeManifest {
                kind: kind.as_str(),
            });
        }
    };
    let manifest: SaeManifest = serde_json::from_value(manifest_value.clone()).map_err(|err| {
        SteeringModuleLoadError::SaeManifest {
            message: format!("{err}"),
        }
    })?;
    for (layer, hook) in &manifest.layers {
        if hook.is_empty() {
            return Err(SteeringModuleLoadError::SaeManifest {
                message: format!("layers entry [{layer}, ...] has an empty hook name"),
            });
        }
    }

    let weights_value = match obj.get("sae_weights") {
        Some(value) if !value.is_null() => value,
        _ => {
            return Err(SteeringModuleLoadError::MissingSaeWeights {
                kind: kind.as_str(),
            });
        }
    };
    let weights: SaeWeights = serde_json::from_value(weights_value.clone()).map_err(|err| {
        SteeringModuleLoadError::SaeWeights {
            site: "<payload>".to_owned(),
            message: format!(
                "must be a map of \"layer:hook\" to tensor-name → {{dtype, shape, data}}: {err}"
            ),
        }
    })?;
    validate_sae_weights(kind, &manifest, &weights)?;

    Ok(SteeringModuleBroadcast {
        kind,
        vectors: None,
        prefill_vectors: None,
        decode_vectors: None,
        sae_manifest: Some(manifest),
        sae_weights: Some(weights),
    })
}

/// Parse the legacy additive `{vectors, prefill_vectors, decode_vectors}`
/// shape (SAE-only fields are rejected).
fn parse_additive_module(
    obj: &serde_json::Map<String, Value>,
) -> Result<SteeringModuleBroadcast, SteeringModuleLoadError> {
    for field in ["sae_manifest", "sae_weights"] {
        if obj.get(field).is_some_and(|v| !v.is_null()) {
            return Err(SteeringModuleLoadError::SaeFieldOnAdditive { field });
        }
    }
    Ok(SteeringModuleBroadcast {
        kind: SteeringModuleKind::Additive,
        vectors: load_tier(obj.get("vectors"), "vectors")?,
        prefill_vectors: load_tier(obj.get("prefill_vectors"), "prefill_vectors")?,
        decode_vectors: load_tier(obj.get("decode_vectors"), "decode_vectors")?,
        sae_manifest: None,
        sae_weights: None,
    })
}

/// Tensor names every SAE site must carry so the worker's attach method
/// (`attach_sae_weights` / `attach_sae_full_recon_weights`) finds the keys it
/// requires. Extra tensor names (e.g. `threshold`) pass through opaquely.
fn required_sae_tensor_keys(kind: SteeringModuleKind) -> &'static [&'static str] {
    match kind {
        SteeringModuleKind::SaeFullReconstruction => &[
            "encoder_weight",
            "encoder_bias",
            "decoder_weight",
            "decoder_bias",
        ],
        _ => &["encoder_weight", "encoder_bias", "decoder_weight"],
    }
}

/// Cheap structural validation of an SAE weights payload: site keys are
/// `"layer:hook"` and cover exactly the manifest's declared sites, each site
/// carries the tensor keys the worker's attach method requires, and every
/// packed blob's base64 decodes to `product(shape) × dtype-itemsize` bytes.
/// Floats are never decoded — the worker rebuilds tensors from the same
/// base64 payload.
fn validate_sae_weights(
    kind: SteeringModuleKind,
    manifest: &SaeManifest,
    weights: &SaeWeights,
) -> Result<(), SteeringModuleLoadError> {
    let site_err = |site: &str, message: String| SteeringModuleLoadError::SaeWeights {
        site: site.to_owned(),
        message,
    };

    let mut provided_sites: HashSet<(u32, String)> = HashSet::with_capacity(weights.len());
    for (site_key, tensors) in weights {
        let (layer_str, hook) = site_key
            .split_once(':')
            .ok_or_else(|| site_err(site_key, "site key must be \"layer:hook\"".to_owned()))?;
        let layer: u32 = layer_str.parse().map_err(|_| {
            site_err(
                site_key,
                format!("site key layer `{layer_str}` is not a non-negative integer"),
            )
        })?;
        if hook.is_empty() {
            return Err(site_err(site_key, "site key hook name is empty".to_owned()));
        }
        provided_sites.insert((layer, hook.to_owned()));

        for required in required_sae_tensor_keys(kind) {
            if !tensors.contains_key(*required) {
                return Err(site_err(
                    site_key,
                    format!(
                        "missing required tensor `{required}` for kind `{}`",
                        kind.as_str()
                    ),
                ));
            }
        }
        for (tensor_name, packed) in tensors {
            let element_size = steering_dtype_element_size(&packed.dtype).map_err(|source| {
                site_err(site_key, format!("tensor `{tensor_name}`: {source}"))
            })?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(packed.data.as_bytes())
                .map_err(|err| {
                    site_err(
                        site_key,
                        format!("tensor `{tensor_name}` data is not valid base64: {err}"),
                    )
                })?;
            let elements: u64 = packed.shape.iter().product();
            let expected = (elements as usize).checked_mul(element_size).ok_or_else(|| {
                site_err(
                    site_key,
                    format!("tensor `{tensor_name}` shape {:?} overflows", packed.shape),
                )
            })?;
            if bytes.len() != expected {
                return Err(site_err(
                    site_key,
                    format!(
                        "tensor `{tensor_name}` byte length {} does not match expected {expected} \
                         for shape {:?} dtype `{}`",
                        bytes.len(),
                        packed.shape,
                        packed.dtype
                    ),
                ));
            }
        }
    }

    // Exact coverage: a manifest site with no weights would leave that
    // site's worker buffers zero-filled (the full-reconstruction attach path
    // skips missing sites silently), and a weights site absent from the
    // manifest is dead payload — both are almost certainly typos.
    let manifest_sites: HashSet<(u32, String)> =
        manifest.layers.iter().map(|(l, h)| (*l, h.clone())).collect();
    for (layer, hook) in &manifest_sites {
        if !provided_sites.contains(&(*layer, hook.clone())) {
            return Err(site_err(
                &format!("{layer}:{hook}"),
                "manifest declares this site but `sae_weights` has no entry for it".to_owned(),
            ));
        }
    }
    for (layer, hook) in &provided_sites {
        if !manifest_sites.contains(&(*layer, hook.clone())) {
            return Err(site_err(
                &format!("{layer}:{hook}"),
                "`sae_weights` carries this site but the manifest does not declare it".to_owned(),
            ));
        }
    }
    Ok(())
}

/// Load all configured modules and broadcast them to the engine workers,
/// returning the set of successfully registered module names.
///
/// On any load or broadcast failure the whole startup fails loudly — a
/// half-registered registry would let some `steering_name` references resolve
/// and others raise at request time.
pub async fn load_and_broadcast_steering_modules(
    client: &EngineCoreClient,
    modules: &[SteeringModulePath],
) -> anyhow::Result<HashSet<String>> {
    if modules.is_empty() {
        return Ok(HashSet::new());
    }

    let mut payload: HashMap<String, SteeringModuleBroadcast> =
        HashMap::with_capacity(modules.len());
    for module in modules {
        let loaded = load_steering_module(&module.path)
            .with_context(|| format!("failed to load steering module '{}'", module.name))?;
        payload.insert(module.name.clone(), loaded);
    }

    // Startup pushes the full registry, replacing any prior state.
    register_modules(client, &payload, true).await?;

    let names: HashSet<String> = payload.into_keys().collect();
    let mut sorted: Vec<&str> = names.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    info!(count = names.len(), modules = ?sorted, "loaded steering modules");
    Ok(names)
}

/// Broadcast a set of modules to every engine worker and pre-materialize them.
///
/// With `replace = true` the worker registry is cleared first (the modules
/// become the entire registry); with `replace = false` they are added to /
/// override the existing entries. Shared by startup and the runtime register
/// endpoint.
pub async fn register_modules(
    client: &EngineCoreClient,
    payload: &HashMap<String, SteeringModuleBroadcast>,
    replace: bool,
) -> anyhow::Result<()> {
    // Worker `kwargs` must reach Python as a msgpack *map*. The utility-call
    // path encodes args with `rmpv::ext::to_value`, which serializes Rust
    // structs as arrays — so the kwargs (and the nested module specs) are built
    // as `serde_json::Value` maps instead. Layer keys become strings here; the
    // worker's `_module_payload_to_specs` coerces them back to int.
    let modules = serde_json::to_value(payload).context("encode steering module payload")?;
    let kwargs = serde_json::json!({ "modules": modules, "replace": replace });
    client
        .collective_rpc(
            "register_steering_modules",
            None,
            Vec::<Value>::new(),
            kwargs,
        )
        .await
        .context("failed to broadcast steering modules to engine workers")?;

    // Eagerly materialize each additive module's rows so the first request
    // resolving to one hits the warm cache instead of paying the cold-path
    // upload. SAE kinds are exempt: they have no precomputed rows to
    // pre-materialize (the worker attaches encoder/decoder buffers during the
    // register RPC above), mirroring the Python modules router's
    // additive-only gate.
    for (name, module) in payload {
        if module.kind.is_sae() {
            continue;
        }
        let kwargs = serde_json::json!({ "name": name });
        client
            .collective_rpc(
                "pre_materialize_steering_module",
                None,
                Vec::<Value>::new(),
                kwargs,
            )
            .await
            .with_context(|| format!("failed to pre-materialize steering module '{name}'"))?;
    }
    Ok(())
}

/// Drop the named modules from every engine worker's registry, releasing the
/// pre-materialized row pins they held.
pub async fn unregister_modules(client: &EngineCoreClient, names: &[String]) -> anyhow::Result<()> {
    if names.is_empty() {
        return Ok(());
    }
    let kwargs = serde_json::json!({ "names": names });
    client
        .collective_rpc(
            "unregister_steering_modules",
            None,
            Vec::<Value>::new(),
            kwargs,
        )
        .await
        .context("failed to unregister steering modules on engine workers")?;
    Ok(())
}

/// Decode one tier value into the inline spec, or `None` when absent/empty.
fn load_tier(
    value: Option<&Value>,
    tier: &'static str,
) -> Result<Option<SteeringVectorSpec>, SteeringModuleLoadError> {
    let obj = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(obj)) => obj,
        Some(_) => {
            return Err(SteeringModuleLoadError::Tier {
                tier,
                message: "tier must be a JSON object keyed by hook point".to_owned(),
            });
        }
    };
    if obj.is_empty() {
        return Ok(None);
    }

    // Detect the packed shape from the first hook value: a packed blob is an
    // object carrying `data` and `dtype` (mirrors Python `_looks_packed`).
    let looks_packed = obj
        .values()
        .next()
        .and_then(Value::as_object)
        .is_some_and(|first| first.contains_key("data") && first.contains_key("dtype"));

    if looks_packed {
        let packed: SteeringSpecPacked = serde_json::from_value(Value::Object(obj.clone()))
            .map_err(|err| SteeringModuleLoadError::Tier {
                tier,
                message: format!("invalid packed steering blob: {err}"),
            })?;
        let spec = unpack_steering_spec(&packed)
            .map_err(|source| SteeringModuleLoadError::Decode { tier, source })?;
        return Ok((!spec.is_empty()).then_some(spec));
    }

    let mut spec = SteeringVectorSpec::with_capacity(obj.len());
    for (hook, layers_value) in obj {
        let layers = layers_value.as_object().ok_or_else(|| SteeringModuleLoadError::Tier {
            tier,
            message: format!("hook '{hook}' must map layer indices to entries"),
        })?;
        let mut layer_map = HashMap::with_capacity(layers.len());
        for (layer_key, entry_value) in layers {
            let layer: u32 = layer_key.parse().map_err(|_| SteeringModuleLoadError::Tier {
                tier,
                message: format!("hook '{hook}' has non-integer layer key '{layer_key}'"),
            })?;
            layer_map.insert(layer, parse_inline_entry(tier, hook, entry_value)?);
        }
        spec.insert(hook.clone(), layer_map);
    }
    Ok(Some(spec))
}

/// Parse one inline layer entry: a bare `[floats]` (scale 1.0) or a
/// `{"vector": [...], "scale": s}` object (scale defaults to 1.0).
fn parse_inline_entry(
    tier: &'static str,
    hook: &str,
    value: &Value,
) -> Result<SteeringLayerEntry, SteeringModuleLoadError> {
    let invalid = |message: String| SteeringModuleLoadError::Tier { tier, message };
    match value {
        Value::Array(items) => Ok(SteeringLayerEntry {
            vector: parse_float_array(items).ok_or_else(|| {
                invalid(format!("hook '{hook}' vector must be a list of numbers"))
            })?,
            scale: 1.0,
        }),
        Value::Object(entry) => {
            let vector_value = entry.get("vector").and_then(Value::as_array).ok_or_else(|| {
                invalid(format!(
                    "hook '{hook}' entry must contain a numeric `vector` list"
                ))
            })?;
            let vector = parse_float_array(vector_value).ok_or_else(|| {
                invalid(format!("hook '{hook}' vector must be a list of numbers"))
            })?;
            let scale = match entry.get("scale") {
                None | Some(Value::Null) => 1.0,
                Some(s) => s
                    .as_f64()
                    .ok_or_else(|| invalid(format!("hook '{hook}' scale must be a number")))?
                    as f32,
            };
            Ok(SteeringLayerEntry { vector, scale })
        }
        _ => Err(invalid(format!(
            "hook '{hook}' entry must be a list or {{vector, scale}} object"
        ))),
    }
}

/// Convert a JSON array of numbers into `Vec<f32>`, or `None` if any element is
/// not numeric.
fn parse_float_array(items: &[Value]) -> Option<Vec<f32>> {
    items.iter().map(|v| v.as_f64().map(|f| f as f32)).collect()
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;

    fn write_temp(contents: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(contents.as_bytes()).expect("write");
        file.flush().expect("flush");
        file
    }

    #[test]
    fn loads_inline_module_with_string_layer_keys() {
        let file = write_temp(
            r#"{
                "vectors": {"post_block": {"14": [0.1, 0.2, 0.3]}},
                "decode_vectors": {"pre_attn": {"3": {"vector": [1.0, 2.0], "scale": 0.5}}}
            }"#,
        );
        let module = load_steering_module(file.path().to_str().unwrap()).expect("load");

        let base = module.vectors.expect("vectors");
        assert_eq!(base["post_block"][&14].vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(base["post_block"][&14].scale, 1.0);
        assert!(module.prefill_vectors.is_none());
        let decode = module.decode_vectors.expect("decode");
        assert_eq!(decode["pre_attn"][&3].vector, vec![1.0, 2.0]);
        assert_eq!(decode["pre_attn"][&3].scale, 0.5);
    }

    #[test]
    fn loads_packed_module_tier() {
        use base64::Engine as _;
        let data = base64::engine::general_purpose::STANDARD
            .encode([1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        let file = write_temp(&format!(
            r#"{{"vectors": {{"pre_attn": {{"dtype": "float32", "shape": [1, 2], "layer_indices": [7], "data": "{data}"}}}}}}"#,
        ));
        let module = load_steering_module(file.path().to_str().unwrap()).expect("load");
        let base = module.vectors.expect("vectors");
        assert_eq!(base["pre_attn"][&7].vector, vec![1.0, 2.0]);
    }

    #[test]
    fn missing_file_is_an_error() {
        assert!(matches!(
            load_steering_module("/nonexistent/steering.json"),
            Err(SteeringModuleLoadError::NotFound { .. })
        ));
    }

    #[test]
    fn non_integer_layer_key_is_rejected() {
        let file = write_temp(r#"{"vectors": {"post_block": {"oops": [0.1]}}}"#);
        assert!(matches!(
            load_steering_module(file.path().to_str().unwrap()),
            Err(SteeringModuleLoadError::Tier { .. })
        ));
    }

    #[test]
    fn additive_broadcast_serializes_kind_and_skips_sae_fields() {
        let file = write_temp(r#"{"vectors": {"post_block": {"14": [0.1, 0.2]}}}"#);
        let module = load_steering_module(file.path().to_str().unwrap()).expect("load");
        assert_eq!(module.kind, SteeringModuleKind::Additive);

        let json = serde_json::to_value(&module).expect("serialize");
        assert_eq!(json["kind"], "additive");
        assert!(json.get("sae_manifest").is_none());
        assert!(json.get("sae_weights").is_none());
        assert!(json.get("vectors").is_some());
    }

    /// Base64 of `n` little-endian f32 values.
    fn b64_f32(values: &[f32]) -> String {
        base64::engine::general_purpose::STANDARD
            .encode(values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
    }

    /// A valid `sae_delta` module JSON string covering site `20:post_block`
    /// with `d_model = 4` and two clampable features.
    fn sae_delta_module_json() -> String {
        format!(
            r#"{{
                "kind": "sae_delta",
                "sae_manifest": {{
                    "d_model": 4,
                    "d_sae": 8,
                    "activation": "relu",
                    "layers": [[20, "post_block"]],
                    "clampable_features": [0, 1],
                    "activation_params": {{}}
                }},
                "sae_weights": {{
                    "20:post_block": {{
                        "encoder_weight": {{"dtype": "float32", "shape": [2, 4], "data": "{ew}"}},
                        "encoder_bias": {{"dtype": "float32", "shape": [2], "data": "{eb}"}},
                        "decoder_weight": {{"dtype": "float32", "shape": [2, 4], "data": "{dw}"}}
                    }}
                }}
            }}"#,
            ew = b64_f32(&[1.0; 8]),
            eb = b64_f32(&[0.5; 2]),
            dw = b64_f32(&[2.0; 8]),
        )
    }

    #[test]
    fn loads_sae_delta_module_and_serializes_broadcast_shape() {
        let file = write_temp(&sae_delta_module_json());
        let module = load_steering_module(file.path().to_str().unwrap()).expect("load");

        assert_eq!(module.kind, SteeringModuleKind::SaeDelta);
        assert!(module.vectors.is_none());
        let manifest = module.sae_manifest.as_ref().expect("manifest");
        assert_eq!(manifest.d_model, 4);
        assert_eq!(manifest.layers, vec![(20, "post_block".to_owned())]);
        let weights = module.sae_weights.as_ref().expect("weights");
        assert_eq!(weights["20:post_block"]["encoder_weight"].shape, vec![2, 4]);

        // The broadcast JSON matches the worker's expected payload shape:
        // kind + sae_manifest (pair-list layers) + "layer:hook"-keyed packed
        // weights, with no additive tier keys.
        let json = serde_json::to_value(&module).expect("serialize");
        assert_eq!(json["kind"], "sae_delta");
        assert_eq!(
            json["sae_manifest"]["layers"],
            serde_json::json!([[20, "post_block"]])
        );
        assert_eq!(json["sae_manifest"]["d_sae"], 8);
        let tensor = &json["sae_weights"]["20:post_block"]["encoder_weight"];
        assert_eq!(tensor["dtype"], "float32");
        assert_eq!(tensor["shape"], serde_json::json!([2, 4]));
        assert!(tensor["data"].is_string());
        assert!(json.get("vectors").is_none());
        assert!(json.get("prefill_vectors").is_none());
    }

    #[test]
    fn sae_weight_byte_length_mismatch_is_rejected() {
        let mut value: Value = serde_json::from_str(&sae_delta_module_json()).unwrap();
        // 8 elements declared, only 2 encoded.
        value["sae_weights"]["20:post_block"]["encoder_weight"]["data"] =
            Value::from(b64_f32(&[1.0, 2.0]));
        let obj = value.as_object().unwrap();
        let err = parse_module(obj).unwrap_err();
        assert!(
            matches!(&err, SteeringModuleLoadError::SaeWeights { site, message }
                if site == "20:post_block" && message.contains("byte length")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sae_weight_invalid_base64_is_rejected() {
        let mut value: Value = serde_json::from_str(&sae_delta_module_json()).unwrap();
        value["sae_weights"]["20:post_block"]["encoder_bias"]["data"] =
            Value::from("!!! not base64 !!!");
        let err = parse_module(value.as_object().unwrap()).unwrap_err();
        assert!(
            matches!(&err, SteeringModuleLoadError::SaeWeights { message, .. }
                if message.contains("base64")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unknown_kind_is_rejected() {
        let value: Value = serde_json::from_str(r#"{"kind": "sae_banana"}"#).unwrap();
        assert!(matches!(
            parse_module(value.as_object().unwrap()),
            Err(SteeringModuleLoadError::UnknownKind { .. })
        ));
    }

    #[test]
    fn sae_kind_with_additive_tier_is_rejected() {
        let mut value: Value = serde_json::from_str(&sae_delta_module_json()).unwrap();
        value["vectors"] = serde_json::json!({"post_block": {"14": [0.1]}});
        assert!(matches!(
            parse_module(value.as_object().unwrap()),
            Err(SteeringModuleLoadError::AdditiveTierOnSae {
                tier: "vectors",
                ..
            })
        ));
    }

    #[test]
    fn additive_kind_with_sae_manifest_is_rejected() {
        let value: Value =
            serde_json::from_str(r#"{"kind": "additive", "sae_manifest": {"d_model": 4}}"#)
                .unwrap();
        assert!(matches!(
            parse_module(value.as_object().unwrap()),
            Err(SteeringModuleLoadError::SaeFieldOnAdditive {
                field: "sae_manifest"
            })
        ));
    }

    #[test]
    fn sae_kind_without_manifest_or_weights_is_rejected() {
        let value: Value = serde_json::from_str(r#"{"kind": "sae_delta"}"#).unwrap();
        assert!(matches!(
            parse_module(value.as_object().unwrap()),
            Err(SteeringModuleLoadError::MissingSaeManifest { kind: "sae_delta" })
        ));

        let mut with_manifest: Value = serde_json::from_str(&sae_delta_module_json()).unwrap();
        with_manifest.as_object_mut().unwrap().remove("sae_weights");
        assert!(matches!(
            parse_module(with_manifest.as_object().unwrap()),
            Err(SteeringModuleLoadError::MissingSaeWeights { kind: "sae_delta" })
        ));
    }

    #[test]
    fn full_reconstruction_requires_decoder_bias() {
        let mut value: Value = serde_json::from_str(&sae_delta_module_json()).unwrap();
        value["kind"] = Value::from("sae_full_reconstruction");
        let err = parse_module(value.as_object().unwrap()).unwrap_err();
        assert!(
            matches!(&err, SteeringModuleLoadError::SaeWeights { message, .. }
                if message.contains("decoder_bias")),
            "unexpected error: {err}"
        );

        value["sae_weights"]["20:post_block"]["decoder_bias"] = serde_json::json!({
            "dtype": "float32", "shape": [4], "data": b64_f32(&[0.0; 4]),
        });
        let module = parse_module(value.as_object().unwrap()).expect("parse");
        assert_eq!(module.kind, SteeringModuleKind::SaeFullReconstruction);
    }

    #[test]
    fn sae_weights_site_coverage_must_match_manifest() {
        // Weights for a site the manifest does not declare.
        let mut value: Value = serde_json::from_str(&sae_delta_module_json()).unwrap();
        let site = value["sae_weights"]["20:post_block"].clone();
        value["sae_weights"]["21:post_block"] = site;
        let err = parse_module(value.as_object().unwrap()).unwrap_err();
        assert!(
            matches!(&err, SteeringModuleLoadError::SaeWeights { site, .. }
                if site == "21:post_block"),
            "unexpected error: {err}"
        );

        // Manifest site with no weights entry.
        let mut value: Value = serde_json::from_str(&sae_delta_module_json()).unwrap();
        value["sae_manifest"]["layers"] =
            serde_json::json!([[20, "post_block"], [21, "post_block"]]);
        let err = parse_module(value.as_object().unwrap()).unwrap_err();
        assert!(
            matches!(&err, SteeringModuleLoadError::SaeWeights { site, .. }
                if site == "21:post_block"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sae_weights_bad_site_key_is_rejected() {
        let mut value: Value = serde_json::from_str(&sae_delta_module_json()).unwrap();
        let weights = value["sae_weights"].as_object_mut().unwrap();
        let site = weights.remove("20:post_block").unwrap();
        weights.insert("post_block".to_owned(), site);
        let err = parse_module(value.as_object().unwrap()).unwrap_err();
        assert!(
            matches!(&err, SteeringModuleLoadError::SaeWeights { site, .. }
                if site == "post_block"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn extra_tensor_names_pass_through() {
        // A per-feature `threshold` tensor (or any future key) rides along
        // untouched as long as the required encoder/decoder keys are present.
        let mut value: Value = serde_json::from_str(&sae_delta_module_json()).unwrap();
        value["sae_weights"]["20:post_block"]["threshold"] = serde_json::json!({
            "dtype": "float32", "shape": [2], "data": b64_f32(&[0.1, 0.2]),
        });
        let module = parse_module(value.as_object().unwrap()).expect("parse");
        let weights = module.sae_weights.expect("weights");
        assert!(weights["20:post_block"].contains_key("threshold"));
    }
}
