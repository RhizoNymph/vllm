//! Decoding of the packed wire format for per-request steering vectors.
//!
//! Northbound clients send steering vectors in a compact binary form (a
//! base64-encoded contiguous tensor plus dtype/shape metadata), matching the
//! Python OpenAI entrypoint's `SteeringHookPacked`. This module decodes that
//! into the inline [`SteeringVectorSpec`] the engine-core resolves, shared by
//! both the HTTP and gRPC frontends.
//!
//! Reference: `vllm/config/steering_types.py` (`unpack_steering_vectors`).

use std::collections::HashMap;

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use vllm_engine_core_client::protocol::{SteeringLayerEntry, SteeringVectorSpec};

/// One hook point's steering blob in the packed wire format.
///
/// Mirrors Python's `SteeringHookPacked` TypedDict: a single contiguous tensor
/// of shape `[num_rows, hidden_size]` where row `i` is the steering vector for
/// layer `layer_indices[i]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SteeringHookPacked {
    /// Numpy dtype string of the encoded data
    /// (`float32` | `float16` | `bfloat16` | `float64`).
    pub dtype: String,
    /// Tensor shape as `[num_rows, hidden_size]`.
    pub shape: Vec<u32>,
    /// Layer index for each row, in row order. Length must equal `shape[0]`.
    pub layer_indices: Vec<u32>,
    /// Base64-encoded contiguous little-endian tensor bytes (row-major).
    pub data: String,
    /// Optional per-row scalar multipliers. Length must equal `shape[0]` when
    /// present; absent means a scale of `1.0` for every row.
    #[serde(default)]
    pub scales: Option<Vec<f32>>,
}

/// A full packed steering spec: hook-point name → packed blob. This is the
/// JSON/proto wire shape of `steering_vectors` / `prefill_steering_vectors` /
/// `decode_steering_vectors`.
pub type SteeringSpecPacked = HashMap<String, SteeringHookPacked>;

/// Errors raised while decoding the packed steering wire format.
#[derive(Debug, thiserror::Error)]
pub enum SteeringDecodeError {
    /// The dtype string is not one of the supported floating-point types.
    #[error(
        "unsupported steering dtype `{dtype}` (expected float32, float16, bfloat16, or float64)"
    )]
    UnsupportedDtype { dtype: String },
    /// `shape` was not a 2-D `[rows, hidden]` pair.
    #[error("steering shape must be [rows, hidden_size], got {shape:?}")]
    InvalidShape { shape: Vec<u32> },
    /// `layer_indices` length did not match the row count.
    #[error("steering layer_indices length {got} does not match row count {rows}")]
    LayerCountMismatch { rows: usize, got: usize },
    /// `scales` length did not match the row count.
    #[error("steering scales length {got} does not match row count {rows}")]
    ScaleCountMismatch { rows: usize, got: usize },
    /// The decoded byte length did not match `shape` × dtype element size.
    #[error(
        "steering data byte length {got} does not match expected {expected} for shape {shape:?} dtype `{dtype}`"
    )]
    ByteLengthMismatch {
        expected: usize,
        got: usize,
        shape: Vec<u32>,
        dtype: String,
    },
    /// The `data` field was not valid base64.
    #[error("steering data is not valid base64: {message}")]
    InvalidBase64 { message: String },
}

/// Supported element encodings for packed steering data.
enum SteeringDtype {
    Float32,
    Float16,
    Bfloat16,
    Float64,
}

impl SteeringDtype {
    fn parse(dtype: &str) -> Result<Self, SteeringDecodeError> {
        match dtype {
            "float32" => Ok(Self::Float32),
            "float16" => Ok(Self::Float16),
            "bfloat16" => Ok(Self::Bfloat16),
            "float64" => Ok(Self::Float64),
            other => Err(SteeringDecodeError::UnsupportedDtype {
                dtype: other.to_owned(),
            }),
        }
    }

    /// Size of one element in bytes.
    fn element_size(&self) -> usize {
        match self {
            Self::Float64 => 8,
            Self::Float32 => 4,
            Self::Float16 | Self::Bfloat16 => 2,
        }
    }

    /// Decode one little-endian element from `bytes` (length == element_size).
    fn decode_element(&self, bytes: &[u8]) -> f32 {
        match self {
            Self::Float32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            Self::Float16 => half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32(),
            Self::Bfloat16 => half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32(),
            Self::Float64 => f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]) as f32,
        }
    }
}

/// Decode one packed hook blob from already-decoded raw bytes into the inline
/// per-layer entries. Shared core for both the HTTP (base64) and gRPC (raw
/// bytes) paths.
pub fn unpack_steering_hook(
    dtype: &str,
    shape: &[u32],
    layer_indices: &[u32],
    scales: Option<&[f32]>,
    data: &[u8],
) -> Result<HashMap<u32, SteeringLayerEntry>, SteeringDecodeError> {
    let parsed = SteeringDtype::parse(dtype)?;

    let [rows, hidden] = *shape else {
        return Err(SteeringDecodeError::InvalidShape {
            shape: shape.to_vec(),
        });
    };
    let rows = rows as usize;
    let hidden = hidden as usize;

    if layer_indices.len() != rows {
        return Err(SteeringDecodeError::LayerCountMismatch {
            rows,
            got: layer_indices.len(),
        });
    }
    if let Some(scales) = scales
        && scales.len() != rows
    {
        return Err(SteeringDecodeError::ScaleCountMismatch {
            rows,
            got: scales.len(),
        });
    }

    let element_size = parsed.element_size();
    let expected = rows * hidden * element_size;
    if data.len() != expected {
        return Err(SteeringDecodeError::ByteLengthMismatch {
            expected,
            got: data.len(),
            shape: shape.to_vec(),
            dtype: dtype.to_owned(),
        });
    }

    let mut entries = HashMap::with_capacity(rows);
    for row in 0..rows {
        let mut vector = Vec::with_capacity(hidden);
        let row_start = row * hidden * element_size;
        for col in 0..hidden {
            let off = row_start + col * element_size;
            vector.push(parsed.decode_element(&data[off..off + element_size]));
        }
        let scale = scales.map_or(1.0, |s| s[row]);
        entries.insert(layer_indices[row], SteeringLayerEntry { vector, scale });
    }
    Ok(entries)
}

/// Decode a full packed steering spec (base64-encoded `data` per hook) into the
/// inline [`SteeringVectorSpec`] forwarded southbound.
pub fn unpack_steering_spec(
    packed: &SteeringSpecPacked,
) -> Result<SteeringVectorSpec, SteeringDecodeError> {
    let mut spec = HashMap::with_capacity(packed.len());
    for (hook, blob) in packed {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(blob.data.as_bytes())
            .map_err(|err| SteeringDecodeError::InvalidBase64 {
                message: format!("{err}"),
            })?;
        let entries = unpack_steering_hook(
            &blob.dtype,
            &blob.shape,
            &blob.layer_indices,
            blob.scales.as_deref(),
            &bytes,
        )?;
        spec.insert(hook.clone(), entries);
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Base64-encode a row-major f32 tensor for fixtures.
    fn encode_f32(rows: &[&[f32]]) -> String {
        let mut bytes = Vec::new();
        for row in rows {
            for v in *row {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    #[test]
    fn unpacks_f32_with_layer_indices_and_default_scale() {
        let packed: SteeringSpecPacked = HashMap::from([(
            "pre_attn".to_string(),
            SteeringHookPacked {
                dtype: "float32".to_string(),
                shape: vec![2, 3],
                layer_indices: vec![5, 9],
                data: encode_f32(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]),
                scales: None,
            },
        )]);

        let spec = unpack_steering_spec(&packed).expect("decode");
        let hook = &spec["pre_attn"];
        assert_eq!(hook[&5].vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(hook[&5].scale, 1.0);
        assert_eq!(hook[&9].vector, vec![4.0, 5.0, 6.0]);
        assert_eq!(hook[&9].scale, 1.0);
    }

    #[test]
    fn applies_per_row_scales() {
        let packed: SteeringSpecPacked = HashMap::from([(
            "post_attn".to_string(),
            SteeringHookPacked {
                dtype: "float32".to_string(),
                shape: vec![1, 2],
                layer_indices: vec![0],
                data: encode_f32(&[&[1.0, 2.0]]),
                scales: Some(vec![0.5]),
            },
        )]);

        let spec = unpack_steering_spec(&packed).expect("decode");
        assert_eq!(spec["post_attn"][&0].scale, 0.5);
    }

    #[test]
    fn decodes_bf16_and_f16() {
        let hidden = [1.0_f32, -2.0];
        let f16_bytes: Vec<u8> =
            hidden.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
        let entries =
            unpack_steering_hook("float16", &[1, 2], &[3], None, &f16_bytes).expect("f16 decode");
        assert_eq!(entries[&3].vector, vec![1.0, -2.0]);

        let bf16_bytes: Vec<u8> =
            hidden.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect();
        let entries = unpack_steering_hook("bfloat16", &[1, 2], &[3], None, &bf16_bytes)
            .expect("bf16 decode");
        assert_eq!(entries[&3].vector, vec![1.0, -2.0]);
    }

    #[test]
    fn rejects_byte_length_mismatch() {
        let err = unpack_steering_hook("float32", &[2, 3], &[0, 1], None, &[0u8; 8]).unwrap_err();
        assert!(matches!(
            err,
            SteeringDecodeError::ByteLengthMismatch {
                expected: 24,
                got: 8,
                ..
            }
        ));
    }

    #[test]
    fn rejects_layer_count_mismatch() {
        let err = unpack_steering_hook("float32", &[2, 1], &[0], None, &[0u8; 8]).unwrap_err();
        assert!(matches!(
            err,
            SteeringDecodeError::LayerCountMismatch { rows: 2, got: 1 }
        ));
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let err = unpack_steering_hook("int8", &[1, 1], &[0], None, &[0u8]).unwrap_err();
        assert!(matches!(err, SteeringDecodeError::UnsupportedDtype { .. }));
    }
}
