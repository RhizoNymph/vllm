//! Parsing of client-submitted directional-clamp tiers into the canonical
//! [`SteeringClamps`] form forwarded southbound.
//!
//! Python types the `SamplingParams` clamp fields as the canonical
//! `SteeringClamps` msgspec Struct and its strict decoder rejects anything
//! else, so this frontend must pack client input HTTP-side. Three input
//! shapes are accepted, matching Python's `SteeringClamps.from_obj`:
//!
//! 1. JSON entry-lists `{hook: {layer: [{vector, min/max|value, strength}]}}`
//!    (string layer keys; `value` sugar pins `min = max`);
//! 2. the legacy base64-packed shape `{hook: {dtype, shape, layer_indices,
//!    data, bounds, strengths}}` (`null` bound = infinite);
//! 3. the canonical type's own JSON form `{"hooks": {hook: {shape,
//!    layer_indices, data: <base64 f64>, lo, hi, strength}}}`.
//!
//! Rows are packed to raw float64 little-endian bytes exactly as Python does
//! (entry-list rows in ascending-layer order, list order within a layer;
//! narrower packed dtypes upcast exactly), so a request submitted through
//! this frontend hashes identically to the same submission through the
//! Python frontend. Validation mirrors Python's ingestion checks so shape
//! errors surface as HTTP 400s here instead of dropped frames at the engine.

use std::collections::BTreeMap;
use std::collections::HashMap;

use base64::Engine as _;
use serde_json::Value;
use vllm_engine_core_client::protocol::{ClampHookTable, SteeringClamps};

/// Errors raised while parsing a clamp tier. Messages intentionally echo the
/// Python-side ingestion errors so both frontends 400 with similar text.
#[derive(Debug, thiserror::Error)]
pub enum ClampParseError {
    #[error("clamp tier must be a JSON object keyed by hook point")]
    TierNotObject,
    #[error("[{hook}] must map layer indices to entry lists")]
    HookNotObject { hook: String },
    #[error("[{hook}] has invalid layer index {key}; expected a non-negative integer")]
    BadLayerKey { hook: String, key: String },
    #[error("[{hook}][{layer}] must be a list of clamp entries")]
    EntriesNotList { hook: String, layer: u32 },
    #[error("[{hook}][{layer}][{index}]: {message}")]
    BadEntry {
        hook: String,
        layer: u32,
        index: usize,
        message: String,
    },
    #[error("[{hook}]: direction width {got} != {expected} of earlier entries in the same hook")]
    WidthMismatch {
        hook: String,
        expected: usize,
        got: usize,
    },
    #[error("[{hook}]: {message}")]
    BadPackedBlob { hook: String, message: String },
    #[error("[{hook}]: clamp direction rows must contain only finite floats")]
    NonFiniteRow { hook: String },
    #[error("[{hook}]: clamp direction rows must be non-zero")]
    ZeroRow { hook: String },
}

/// One parsed clamp entry, pre-packing.
struct ParsedEntry {
    vector: Vec<f64>,
    lo: f64,
    hi: f64,
    strength: f64,
}

/// Parse one optional clamp tier `Value` into the canonical form. `None`,
/// JSON `null` and empty objects collapse to `None` (matching Python's
/// `from_obj` empty collapse).
pub fn parse_clamp_tier(value: Option<&Value>) -> Result<Option<SteeringClamps>, ClampParseError> {
    let obj = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(obj)) => obj,
        Some(_) => return Err(ClampParseError::TierNotObject),
    };
    if obj.is_empty() {
        return Ok(None);
    }

    // The canonical type's own JSON form: a single "hooks" key ("hooks" is
    // not a valid hook-point name, so this is unambiguous).
    if obj.len() == 1
        && let Some(hooks) = obj.get("hooks")
    {
        return parse_wire_map(hooks);
    }

    // Legacy base64-packed shape: marker keys data + dtype + bounds on the
    // first hook blob (mirrors Python's `_clamps_looks_packed`).
    if let Some(Value::Object(first)) = obj.values().next()
        && first.contains_key("data")
        && first.contains_key("dtype")
        && first.contains_key("bounds")
    {
        return parse_legacy_packed(obj);
    }

    parse_entry_lists(obj)
}

/// Parse the entry-list authoring shape, packing rows at float64 in
/// ascending-layer order (list order within a layer) — byte-for-byte the
/// order Python's `from_obj` packs, so hashes match across frontends.
fn parse_entry_lists(
    obj: &serde_json::Map<String, Value>,
) -> Result<Option<SteeringClamps>, ClampParseError> {
    let mut hooks = HashMap::with_capacity(obj.len());
    for (hook, layers_value) in obj {
        let Value::Object(layers_obj) = layers_value else {
            return Err(ClampParseError::HookNotObject { hook: hook.clone() });
        };
        // BTreeMap gives the ascending-layer packing order.
        let mut layers: BTreeMap<u32, &Vec<Value>> = BTreeMap::new();
        for (key, entries_value) in layers_obj {
            let layer: u32 = key.parse().map_err(|_| ClampParseError::BadLayerKey {
                hook: hook.clone(),
                key: key.clone(),
            })?;
            let Value::Array(entries) = entries_value else {
                return Err(ClampParseError::EntriesNotList {
                    hook: hook.clone(),
                    layer,
                });
            };
            layers.insert(layer, entries);
        }

        let mut data: Vec<u8> = Vec::new();
        let mut layer_indices: Vec<u32> = Vec::new();
        let mut lo: Vec<f64> = Vec::new();
        let mut hi: Vec<f64> = Vec::new();
        let mut strength: Vec<f64> = Vec::new();
        let mut width: Option<usize> = None;
        for (&layer, entries) in &layers {
            for (index, entry) in entries.iter().enumerate() {
                let parsed = parse_entry(entry).map_err(|message| ClampParseError::BadEntry {
                    hook: hook.clone(),
                    layer,
                    index,
                    message,
                })?;
                match width {
                    None => width = Some(parsed.vector.len()),
                    Some(expected) if expected != parsed.vector.len() => {
                        return Err(ClampParseError::WidthMismatch {
                            hook: hook.clone(),
                            expected,
                            got: parsed.vector.len(),
                        });
                    }
                    Some(_) => {}
                }
                for v in &parsed.vector {
                    data.extend_from_slice(&v.to_le_bytes());
                }
                layer_indices.push(layer);
                lo.push(parsed.lo);
                hi.push(parsed.hi);
                strength.push(parsed.strength);
            }
        }
        if layer_indices.is_empty() {
            continue;
        }
        let width = width.expect("rows imply width") as u32;
        hooks.insert(
            hook.clone(),
            ClampHookTable {
                shape: vec![layer_indices.len() as u32, width],
                layer_indices,
                data,
                lo,
                hi,
                strength,
            },
        );
    }
    Ok((!hooks.is_empty()).then_some(SteeringClamps { hooks }))
}

/// Validate and resolve one entry object — the exact semantics of Python's
/// `_parse_clamp_entry` (`value` sugar, omitted bounds → `±inf`, strength
/// default 1.0, finite non-zero vector).
fn parse_entry(entry: &Value) -> Result<ParsedEntry, String> {
    let Value::Object(map) = entry else {
        return Err(format!(
            "Clamp entry must be a dict, got {}",
            type_name(entry)
        ));
    };
    const ALLOWED: [&str; 5] = ["vector", "value", "min", "max", "strength"];
    let mut extra: Vec<&str> =
        map.keys().map(String::as_str).filter(|k| !ALLOWED.contains(k)).collect();
    if !extra.is_empty() {
        extra.sort_unstable();
        return Err(format!("Clamp entry has unexpected keys: {extra:?}"));
    }

    let get = |key: &str| map.get(key).filter(|v| !v.is_null());
    let as_float = |key: &str| -> Result<Option<f64>, String> {
        get(key)
            .map(|v| v.as_f64().ok_or_else(|| format!("Clamp entry '{key}' must be a number")))
            .transpose()
    };

    let value = as_float("value")?;
    let min = as_float("min")?;
    let max = as_float("max")?;
    let (lo, hi) = match (value, min, max) {
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
            return Err("Clamp entry 'value' is mutually exclusive with 'min'/'max'".to_owned());
        }
        (Some(v), None, None) => (v, v),
        (None, None, None) => {
            return Err("Clamp entry must set at least one of 'value', 'min', 'max'".to_owned());
        }
        (None, lo, hi) => (lo.unwrap_or(f64::NEG_INFINITY), hi.unwrap_or(f64::INFINITY)),
    };
    if lo > hi {
        return Err(format!("Clamp entry min ({lo}) must be <= max ({hi})"));
    }

    let strength = as_float("strength")?.unwrap_or(1.0);
    if !(0.0..=1.0).contains(&strength) {
        return Err(format!(
            "Clamp entry strength must be in [0, 1], got {strength}"
        ));
    }

    let Some(Value::Array(raw)) = get("vector") else {
        return Err("Clamp entry missing required key 'vector'".to_owned());
    };
    if raw.is_empty() {
        return Err("Clamp entry vector must be a non-empty 1-D float list".to_owned());
    }
    let mut vector = Vec::with_capacity(raw.len());
    for v in raw {
        let f = v
            .as_f64()
            .ok_or_else(|| "Clamp entry vector must contain only finite floats".to_owned())?;
        if !f.is_finite() {
            return Err("Clamp entry vector must contain only finite floats".to_owned());
        }
        vector.push(f);
    }
    if vector.iter().all(|v| *v == 0.0) {
        return Err("Clamp entry vector must be non-zero".to_owned());
    }
    Ok(ParsedEntry {
        vector,
        lo,
        hi,
        strength,
    })
}

/// Parse the legacy base64-packed input shape, upcasting rows exactly to
/// float64.
fn parse_legacy_packed(
    obj: &serde_json::Map<String, Value>,
) -> Result<Option<SteeringClamps>, ClampParseError> {
    let mut hooks = HashMap::with_capacity(obj.len());
    for (hook, blob_value) in obj {
        let bad = |message: String| ClampParseError::BadPackedBlob {
            hook: hook.clone(),
            message,
        };
        let Value::Object(blob) = blob_value else {
            return Err(bad("packed clamp blob must be a JSON object".to_owned()));
        };
        let dtype = blob
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("missing `dtype`".to_owned()))?;
        let shape = parse_u32_array(blob.get("shape")).map_err(|m| bad(format!("shape: {m}")))?;
        let [rows, hidden] = shape[..] else {
            return Err(bad(format!("shape must be [n, hidden]; got {shape:?}")));
        };
        let n = rows as usize;
        let layer_indices = parse_u32_array(blob.get("layer_indices"))
            .map_err(|m| bad(format!("layer_indices: {m}")))?;
        if layer_indices.len() != n {
            return Err(bad(format!(
                "layer_indices length {} must equal shape[0] {n}",
                layer_indices.len()
            )));
        }
        let data_b64 = blob
            .get("data")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("missing base64 `data`".to_owned()))?;
        let raw = base64::engine::general_purpose::STANDARD
            .decode(data_b64.as_bytes())
            .map_err(|err| bad(format!("data is not valid base64: {err}")))?;
        let data = upcast_rows_to_f64_le(dtype, n, hidden as usize, &raw).map_err(&bad)?;

        let Some(Value::Array(bounds)) = blob.get("bounds") else {
            return Err(bad("missing `bounds` list".to_owned()));
        };
        if bounds.len() != n {
            return Err(bad(format!(
                "bounds length {} must equal shape[0] {n}",
                bounds.len()
            )));
        }
        let mut lo = Vec::with_capacity(n);
        let mut hi = Vec::with_capacity(n);
        for pair in bounds {
            let Value::Array(pair) = pair else {
                return Err(bad("each bound must be a [lo, hi] pair".to_owned()));
            };
            let [lo_raw, hi_raw] = &pair[..] else {
                return Err(bad("each bound must be a [lo, hi] pair".to_owned()));
            };
            lo.push(parse_bound(lo_raw, f64::NEG_INFINITY).map_err(&bad)?);
            hi.push(parse_bound(hi_raw, f64::INFINITY).map_err(&bad)?);
        }

        let strength =
            parse_f64_array(blob.get("strengths")).map_err(|m| bad(format!("strengths: {m}")))?;
        if strength.len() != n {
            return Err(bad(format!(
                "strengths length {} must equal shape[0] {n}",
                strength.len()
            )));
        }

        let table = ClampHookTable {
            shape: vec![rows, hidden],
            layer_indices,
            data,
            lo,
            hi,
            strength,
        };
        check_rows(hook, &table)?;
        hooks.insert(hook.clone(), table);
    }
    Ok((!hooks.is_empty()).then_some(SteeringClamps { hooks }))
}

/// Parse the canonical type's own JSON form (base64 `data`, `null` = `±inf`).
fn parse_wire_map(hooks_value: &Value) -> Result<Option<SteeringClamps>, ClampParseError> {
    let Value::Object(hooks_obj) = hooks_value else {
        return Err(ClampParseError::TierNotObject);
    };
    let mut hooks = HashMap::with_capacity(hooks_obj.len());
    for (hook, blob_value) in hooks_obj {
        let bad = |message: String| ClampParseError::BadPackedBlob {
            hook: hook.clone(),
            message,
        };
        let Value::Object(blob) = blob_value else {
            return Err(bad("clamp hook table must be a JSON object".to_owned()));
        };
        let shape = parse_u32_array(blob.get("shape")).map_err(|m| bad(format!("shape: {m}")))?;
        let [rows, hidden] = shape[..] else {
            return Err(bad(format!("shape must be [n, hidden]; got {shape:?}")));
        };
        let n = rows as usize;
        let layer_indices = parse_u32_array(blob.get("layer_indices"))
            .map_err(|m| bad(format!("layer_indices: {m}")))?;
        let data_b64 = blob
            .get("data")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("missing base64 `data`".to_owned()))?;
        let data = base64::engine::general_purpose::STANDARD
            .decode(data_b64.as_bytes())
            .map_err(|err| bad(format!("data is not valid base64: {err}")))?;
        if data.len() != n * hidden as usize * 8 {
            return Err(bad(format!(
                "data length {} != expected {} (shape [{rows}, {hidden}], float64)",
                data.len(),
                n * hidden as usize * 8
            )));
        }
        let parse_bounds = |key: &str, open: f64| -> Result<Vec<f64>, ClampParseError> {
            let Some(Value::Array(vals)) = blob.get(key) else {
                return Err(bad(format!("missing `{key}` list")));
            };
            vals.iter().map(|v| parse_bound(v, open).map_err(&bad)).collect()
        };
        let lo = parse_bounds("lo", f64::NEG_INFINITY)?;
        let hi = parse_bounds("hi", f64::INFINITY)?;
        let strength =
            parse_f64_array(blob.get("strength")).map_err(|m| bad(format!("strength: {m}")))?;
        for (name, len) in [
            ("layer_indices", layer_indices.len()),
            ("lo", lo.len()),
            ("hi", hi.len()),
            ("strength", strength.len()),
        ] {
            if len != n {
                return Err(bad(format!("{name} length {len} must equal shape[0] {n}")));
            }
        }
        let table = ClampHookTable {
            shape: vec![rows, hidden],
            layer_indices,
            data,
            lo,
            hi,
            strength,
        };
        check_rows(hook, &table)?;
        hooks.insert(hook.clone(), table);
    }
    Ok((!hooks.is_empty()).then_some(SteeringClamps { hooks }))
}

/// Reject non-finite or all-zero direction rows (Python validates the same
/// eagerly at ingestion).
fn check_rows(hook: &str, table: &ClampHookTable) -> Result<(), ClampParseError> {
    let hidden = table.shape[1] as usize;
    for row in 0..table.shape[0] as usize {
        let mut any_nonzero = false;
        for col in 0..hidden {
            let off = (row * hidden + col) * 8;
            let v = f64::from_le_bytes(table.data[off..off + 8].try_into().expect("8 bytes"));
            if !v.is_finite() {
                return Err(ClampParseError::NonFiniteRow {
                    hook: hook.to_owned(),
                });
            }
            if v != 0.0 {
                any_nonzero = true;
            }
        }
        if !any_nonzero {
            return Err(ClampParseError::ZeroRow {
                hook: hook.to_owned(),
            });
        }
    }
    Ok(())
}

/// Exactly upcast little-endian rows of `dtype` to float64 little-endian.
pub fn upcast_rows_to_f64_le(
    dtype: &str,
    rows: usize,
    hidden: usize,
    raw: &[u8],
) -> Result<Vec<u8>, String> {
    let element_size = match dtype {
        "float64" => 8,
        "float32" => 4,
        "float16" | "bfloat16" => 2,
        other => {
            return Err(format!(
                "unsupported clamp dtype `{other}` (expected float32, float16, bfloat16, or float64)"
            ));
        }
    };
    let expected = rows * hidden * element_size;
    if raw.len() != expected {
        return Err(format!(
            "data length {} != expected {expected} (shape [{rows}, {hidden}], dtype `{dtype}`)",
            raw.len()
        ));
    }
    if dtype == "float64" {
        return Ok(raw.to_vec());
    }
    let mut out = Vec::with_capacity(rows * hidden * 8);
    for chunk in raw.chunks_exact(element_size) {
        // f16/bf16 -> f32 and f32 -> f64 are exact widenings.
        let v: f64 = match dtype {
            "float32" => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64,
            "float16" => half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f64(),
            "bfloat16" => half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f64(),
            _ => unreachable!(),
        };
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

fn parse_bound(value: &Value, open: f64) -> Result<f64, String> {
    match value {
        Value::Null => Ok(open),
        _ => value.as_f64().ok_or_else(|| "bounds must be numbers or null".to_owned()),
    }
}

fn parse_u32_array(value: Option<&Value>) -> Result<Vec<u32>, String> {
    let Some(Value::Array(vals)) = value else {
        return Err("must be a list of non-negative integers".to_owned());
    };
    vals.iter()
        .map(|v| {
            v.as_u64()
                .and_then(|u| u32::try_from(u).ok())
                .ok_or_else(|| "must be a list of non-negative integers".to_owned())
        })
        .collect()
}

fn parse_f64_array(value: Option<&Value>) -> Result<Vec<f64>, String> {
    let Some(Value::Array(vals)) = value else {
        return Err("must be a list of numbers".to_owned());
    };
    vals.iter()
        .map(|v| v.as_f64().ok_or_else(|| "must be a list of numbers".to_owned()))
        .collect()
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "list",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn f64_rows(data: &[u8]) -> Vec<f64> {
        data.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn entry_lists_pack_sorted_layers_and_sugar() {
        let tier = json!({
            "post_block": {
                "9": [{"vector": [0.0, 1.0], "value": 2.0}],
                "3": [{"vector": [3.0, 4.0], "min": -0.5, "max": 1.25}],
            }
        });
        let spec = parse_clamp_tier(Some(&tier)).expect("parse").expect("some");
        let table = &spec.hooks["post_block"];
        assert_eq!(table.shape, vec![2, 2]);
        // Ascending layer order: layer 3's row first.
        assert_eq!(table.layer_indices, vec![3, 9]);
        assert_eq!(f64_rows(&table.data), vec![3.0, 4.0, 0.0, 1.0]);
        assert_eq!(table.lo, vec![-0.5, 2.0]);
        assert_eq!(table.hi, vec![1.25, 2.0]);
        assert_eq!(table.strength, vec![1.0, 1.0]);
    }

    #[test]
    fn omitted_bounds_become_infinite() {
        let tier = json!({"post_block": {"0": [
            {"vector": [1.0], "max": 4.0},
            {"vector": [2.0], "min": -1.0, "strength": 0.5},
        ]}});
        let spec = parse_clamp_tier(Some(&tier)).unwrap().unwrap();
        let table = &spec.hooks["post_block"];
        assert_eq!(table.lo, vec![f64::NEG_INFINITY, -1.0]);
        assert_eq!(table.hi, vec![4.0, f64::INFINITY]);
        assert_eq!(table.strength, vec![1.0, 0.5]);
    }

    #[test]
    fn empty_and_null_collapse_to_none() {
        assert!(parse_clamp_tier(None).unwrap().is_none());
        assert!(parse_clamp_tier(Some(&Value::Null)).unwrap().is_none());
        assert!(parse_clamp_tier(Some(&json!({}))).unwrap().is_none());
        assert!(parse_clamp_tier(Some(&json!({"post_block": {"3": []}}))).unwrap().is_none());
    }

    #[test]
    fn entry_validation_mirrors_python() {
        let bad = |entry: Value| {
            parse_clamp_tier(Some(&json!({"post_block": {"3": [entry]}}))).unwrap_err()
        };
        assert!(format!("{}", bad(json!({"vector": [1.0]}))).contains("at least one"));
        assert!(
            format!(
                "{}",
                bad(json!({"vector": [1.0], "value": 1.0, "max": 2.0}))
            )
            .contains("mutually exclusive")
        );
        assert!(
            format!("{}", bad(json!({"vector": [1.0], "min": 2.0, "max": 1.0})))
                .contains("must be <= max")
        );
        assert!(
            format!("{}", bad(json!({"vector": [0.0, 0.0], "value": 1.0}))).contains("non-zero")
        );
        assert!(
            format!(
                "{}",
                bad(json!({"vector": [1.0], "value": 1.0, "strength": 1.5}))
            )
            .contains("strength")
        );
        assert!(
            format!(
                "{}",
                bad(json!({"vector": [1.0], "value": 1.0, "scale": 2.0}))
            )
            .contains("unexpected keys")
        );
        assert!(format!("{}", bad(json!({"value": 1.0}))).contains("vector"));
    }

    #[test]
    fn bad_layer_key_and_width_mismatch_rejected() {
        let err = parse_clamp_tier(Some(
            &json!({"post_block": {"x": [{"vector": [1.0], "value": 1.0}]}}),
        ))
        .unwrap_err();
        assert!(format!("{err}").contains("invalid layer index"));

        let err = parse_clamp_tier(Some(&json!({"post_block": {
            "3": [{"vector": [1.0, 2.0], "value": 1.0}],
            "4": [{"vector": [1.0], "value": 1.0}],
        }})))
        .unwrap_err();
        assert!(format!("{err}").contains("width"));
    }

    #[test]
    fn legacy_packed_upcasts_to_f64() {
        let rows_f32: Vec<u8> =
            [1.5f32, -2.0, 0.25, 8.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let tier = json!({"post_attn": {
            "dtype": "float32",
            "shape": [2, 2],
            "layer_indices": [3, 5],
            "data": base64::engine::general_purpose::STANDARD.encode(&rows_f32),
            "bounds": [[-2.0, 2.0], [null, 4.0]],
            "strengths": [1.0, 0.5],
        }});
        let spec = parse_clamp_tier(Some(&tier)).unwrap().unwrap();
        let table = &spec.hooks["post_attn"];
        assert_eq!(table.shape, vec![2, 2]);
        assert_eq!(f64_rows(&table.data), vec![1.5, -2.0, 0.25, 8.0]);
        assert_eq!(table.lo, vec![-2.0, f64::NEG_INFINITY]);
        assert_eq!(table.hi, vec![2.0, 4.0]);
        assert_eq!(table.strength, vec![1.0, 0.5]);
    }

    #[test]
    fn legacy_packed_length_mismatch_rejected() {
        let tier = json!({"post_attn": {
            "dtype": "float64",
            "shape": [1, 2],
            "layer_indices": [3],
            "data": base64::engine::general_purpose::STANDARD.encode([0u8; 8]),
            "bounds": [[0.0, 1.0]],
            "strengths": [1.0],
        }});
        let err = parse_clamp_tier(Some(&tier)).unwrap_err();
        assert!(format!("{err}").contains("data length"));
    }

    #[test]
    fn wire_map_round_trips() {
        let rows: Vec<u8> = [1.0f64, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let tier = json!({"hooks": {"post_block": {
            "shape": [1, 2],
            "layer_indices": [14],
            "data": base64::engine::general_purpose::STANDARD.encode(&rows),
            "lo": [null],
            "hi": [2.0],
            "strength": [1.0],
        }}});
        let spec = parse_clamp_tier(Some(&tier)).unwrap().unwrap();
        let table = &spec.hooks["post_block"];
        assert_eq!(table.layer_indices, vec![14]);
        assert_eq!(table.lo, vec![f64::NEG_INFINITY]);
        assert_eq!(table.hi, vec![2.0]);
    }

    #[test]
    fn zero_row_rejected_in_packed_input() {
        let rows: Vec<u8> = [0.0f64, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let tier = json!({"post_attn": {
            "dtype": "float64",
            "shape": [1, 2],
            "layer_indices": [3],
            "data": base64::engine::general_purpose::STANDARD.encode(&rows),
            "bounds": [[0.0, 1.0]],
            "strengths": [1.0],
        }});
        let err = parse_clamp_tier(Some(&tier)).unwrap_err();
        assert!(format!("{err}").contains("non-zero"));
    }
}
