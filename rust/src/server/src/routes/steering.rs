//! Runtime management endpoints for named steering modules.
//!
//! These let clients register, replace, list, and remove named steering modules
//! while the server is running, re-broadcasting the registry to the engine
//! workers — the runtime counterpart of the startup `--steering-modules` load.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, State};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ApiError;
use crate::state::AppState;
use crate::steering_modules::{self, SteeringModuleBroadcast};

/// Request body for `POST /v1/steering/modules`.
#[derive(Debug, Deserialize)]
pub(crate) struct RegisterModulesRequest {
    /// Map of module name → `{vectors, prefill_vectors, decode_vectors}`. Each
    /// tier may be inline (`{hook: {layer: [floats] | {vector, scale}}}`) or
    /// packed (`{hook: {dtype, shape, layer_indices, data, scales}}`).
    modules: serde_json::Map<String, Value>,
    /// When `true`, the provided set becomes the entire registry (existing
    /// modules not listed are dropped). When `false` (default), the modules are
    /// added to / override the existing registry.
    #[serde(default)]
    replace: bool,
}

/// Response body listing the currently registered module names.
#[derive(Debug, Serialize)]
pub(crate) struct SteeringModulesResponse {
    modules: Vec<String>,
}

/// `GET /v1/steering/modules` — list the registered module names.
pub async fn list_steering_modules(
    State(state): State<Arc<AppState>>,
) -> Json<SteeringModulesResponse> {
    Json(SteeringModulesResponse {
        modules: state.list_steering_modules(),
    })
}

/// `POST /v1/steering/modules` — register or replace named modules and
/// re-broadcast the registry to the engine workers.
pub async fn register_steering_modules(
    State(state): State<Arc<AppState>>,
    body: Result<Json<RegisterModulesRequest>, JsonRejection>,
) -> Result<Json<SteeringModulesResponse>, ApiError> {
    let Json(body) = body.map_err(|error| ApiError::json_parse_error(error.body_text()))?;
    if body.modules.is_empty() {
        return Err(ApiError::invalid_request(
            "`modules` must not be empty".to_string(),
            Some("modules"),
        ));
    }

    let mut payload: HashMap<String, SteeringModuleBroadcast> =
        HashMap::with_capacity(body.modules.len());
    for (name, value) in &body.modules {
        let obj = value.as_object().ok_or_else(|| {
            ApiError::invalid_request(
                format!(
                    "module '{name}' must be an object with vectors/prefill_vectors/decode_vectors"
                ),
                Some("modules"),
            )
        })?;
        let parsed = steering_modules::parse_module(obj).map_err(|error| {
            ApiError::invalid_request(format!("module '{name}': {error}"), Some("modules"))
        })?;
        payload.insert(name.clone(), parsed);
    }

    // Serialize against concurrent registry mutations so the broadcast and the
    // local name-set update stay consistent.
    let _guard = state.lock_steering_mutations().await;
    steering_modules::register_modules(state.engine_core_client(), &payload, body.replace)
        .await
        .map_err(|error| ApiError::server_error(format!("{error:#}")))?;

    if body.replace {
        state.set_steering_module_names(payload.keys().cloned().collect());
    } else {
        state.extend_steering_module_names(payload.keys().cloned());
    }

    Ok(Json(SteeringModulesResponse {
        modules: state.list_steering_modules(),
    }))
}

/// `DELETE /v1/steering/modules/{name}` — unregister one module.
pub async fn unregister_steering_module(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<SteeringModulesResponse>, ApiError> {
    let _guard = state.lock_steering_mutations().await;
    if !state.is_steering_module_registered(&name) {
        return Err(ApiError::invalid_request(
            format!("Unknown steering module '{name}'"),
            Some("name"),
        ));
    }
    steering_modules::unregister_modules(state.engine_core_client(), std::slice::from_ref(&name))
        .await
        .map_err(|error| ApiError::server_error(format!("{error:#}")))?;
    state.remove_steering_module_name(&name);

    Ok(Json(SteeringModulesResponse {
        modules: state.list_steering_modules(),
    }))
}
