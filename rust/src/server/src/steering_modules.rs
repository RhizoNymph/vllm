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
use serde::Serialize;
use serde_json::Value;
use tracing::info;
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::{SteeringLayerEntry, SteeringVectorSpec};

use crate::config::SteeringModulePath;
use crate::routes::openai::utils::steering::{
    SteeringDecodeError, SteeringSpecPacked, unpack_steering_spec,
};

/// Broadcast payload for one named module: the three tier specs in the inline
/// form the worker resolves. Field names match the keys read by the worker's
/// `register_steering_modules` (Python `dump_for_broadcast`).
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct SteeringModuleBroadcast {
    /// Base vectors applied to both prefill and decode.
    pub vectors: Option<SteeringVectorSpec>,
    /// Prefill-only additions.
    pub prefill_vectors: Option<SteeringVectorSpec>,
    /// Decode-only additions.
    pub decode_vectors: Option<SteeringVectorSpec>,
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
}

/// Load a single named steering module from its JSON file.
///
/// Each of the `vectors`, `prefill_vectors`, and `decode_vectors` tiers may be
/// either the inline shape (`{hook: {layer: [floats] | {vector, scale}}}`,
/// layer keys as strings or ints) or the packed wire shape (per-hook
/// `{dtype, shape, layer_indices, data, scales}`), detected per tier.
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

/// Parse one module's `{vectors, prefill_vectors, decode_vectors}` object into
/// the broadcast form. Shared by the file loader and the runtime register
/// endpoint (which receives the same shape inline in the request body).
pub fn parse_module(
    obj: &serde_json::Map<String, Value>,
) -> Result<SteeringModuleBroadcast, SteeringModuleLoadError> {
    Ok(SteeringModuleBroadcast {
        vectors: load_tier(obj.get("vectors"), "vectors")?,
        prefill_vectors: load_tier(obj.get("prefill_vectors"), "prefill_vectors")?,
        decode_vectors: load_tier(obj.get("decode_vectors"), "decode_vectors")?,
    })
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

    // Eagerly materialize each module's rows so the first request resolving to
    // one hits the warm cache instead of paying the cold-path upload.
    for name in payload.keys() {
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
                "vectors": {"post_mlp": {"14": [0.1, 0.2, 0.3]}},
                "decode_vectors": {"pre_attn": {"3": {"vector": [1.0, 2.0], "scale": 0.5}}}
            }"#,
        );
        let module = load_steering_module(file.path().to_str().unwrap()).expect("load");

        let base = module.vectors.expect("vectors");
        assert_eq!(base["post_mlp"][&14].vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(base["post_mlp"][&14].scale, 1.0);
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
        let file = write_temp(r#"{"vectors": {"post_mlp": {"oops": [0.1]}}}"#);
        assert!(matches!(
            load_steering_module(file.path().to_str().unwrap()),
            Err(SteeringModuleLoadError::Tier { .. })
        ));
    }
}
