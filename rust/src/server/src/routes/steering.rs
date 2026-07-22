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
use axum::http::HeaderMap;
use axum::http::header::AUTHORIZATION;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ApiError;
use crate::middleware;
use crate::state::AppState;
use crate::steering_modules::{self, SteeringModuleBroadcast};

/// Reject a mutation when steering API keys are configured and the request's
/// bearer token doesn't match — mirroring the Python frontend's
/// `--steering-api-key` gate. No keys configured means unauthenticated.
fn authorize_steering_mutation(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    if state.has_steering_api_keys()
        && !middleware::verify_token(headers.get(AUTHORIZATION), state.steering_api_key_hashes())
    {
        return Err(ApiError::unauthorized(
            "steering API key required".to_string(),
        ));
    }
    Ok(())
}

/// Invalidate KV blocks whose steering hash only names a module.
///
/// Request hashes include named-module references by name/scale, not by the
/// current vector payload or SAE weights, so re-registering a name can make
/// stale prefix-cache blocks appear reusable. Mirrors the Python frontend's
/// `_reset_prefix_cache_after_module_change`: called after every successful
/// register/unregister broadcast (all kinds). The startup load path needs no
/// reset — the cache is empty before serving starts.
async fn reset_prefix_cache_after_module_change(
    state: &AppState,
    action: &'static str,
) -> Result<(), ApiError> {
    match state.engine_core_client().reset_prefix_cache(true, false).await {
        Ok(true) => Ok(()),
        Ok(false) => Err(ApiError::service_unavailable(format!(
            "Steering module was {action} but prefix cache could not be fully invalidated. \
             Retry the request or reset the prefix cache before generating with this module."
        ))),
        Err(error) => Err(ApiError::service_unavailable(format!(
            "Steering module was {action} but the prefix cache reset failed: {error:#}. \
             Reset the prefix cache before generating with this module."
        ))),
    }
}

/// Request body for `POST /v1/steering/modules`.
#[derive(Debug, Deserialize)]
pub(crate) struct RegisterModulesRequest {
    /// Map of module name → module payload. Additive modules carry
    /// `{vectors, prefill_vectors, decode_vectors}` tiers (inline
    /// `{hook: {layer: [floats] | {vector, scale}}}` or packed
    /// `{hook: {dtype, shape, layer_indices, data, scales}}`); SAE modules
    /// (`kind` of `sae_delta` / `sae_full_reconstruction`) carry
    /// `sae_manifest` plus `"layer:hook"`-keyed packed `sae_weights`.
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
    headers: HeaderMap,
    body: Result<Json<RegisterModulesRequest>, JsonRejection>,
) -> Result<Json<SteeringModulesResponse>, ApiError> {
    authorize_steering_mutation(&state, &headers)?;
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

    let registered_kinds =
        payload.iter().map(|(name, module)| (name.clone(), module.kind));
    if body.replace {
        state.set_steering_module_names(registered_kinds.collect());
    } else {
        state.extend_steering_module_names(registered_kinds);
    }

    // The modules are registered at this point even if the reset fails; the
    // 503 tells the client the cache may still serve blocks computed against
    // a previous payload registered under the same name.
    reset_prefix_cache_after_module_change(&state, "registered").await?;

    Ok(Json(SteeringModulesResponse {
        modules: state.list_steering_modules(),
    }))
}

/// `DELETE /v1/steering/modules/{name}` — unregister one module.
pub async fn unregister_steering_module(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    headers: HeaderMap,
) -> Result<Json<SteeringModulesResponse>, ApiError> {
    authorize_steering_mutation(&state, &headers)?;
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

    reset_prefix_cache_after_module_change(&state, "unregistered").await?;

    Ok(Json(SteeringModulesResponse {
        modules: state.list_steering_modules(),
    }))
}
