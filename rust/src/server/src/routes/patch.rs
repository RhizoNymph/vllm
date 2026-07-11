//! Reverse-proxy routes for the activation-patching sweep surface.
//!
//! The Rust frontend does not implement sweep orchestration itself. When a
//! Python patch sidecar (a loopback api_server on the same engines) is
//! configured, these handlers forward `POST /v1/patch_sweep` and
//! `DELETE /v1/patch_source/{run_id}` to it and stream the response back
//! unbuffered — the sweep endpoint can return a `text/event-stream` (SSE)
//! response whose chunks must reach the client as they land. Dropping the
//! upstream connection on client disconnect (which happens automatically when
//! the streamed body is dropped) lets the Python side cancel in-flight GPU
//! work. When no sidecar is configured the routes return HTTP 501.

use std::sync::Arc;

use axum::Json;
use axum::body::Body;
use axum::extract::{Path, Request, State};
use axum::http::{HeaderMap, HeaderName, Method, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::state::{AppState, PatchSidecar};

/// Upper bound on a proxied request body. Sweep requests are small JSON grids;
/// this only guards against unbounded buffering of a hostile body.
const MAX_REQUEST_BODY_BYTES: usize = 32 * 1024 * 1024;

/// Request headers forwarded upstream to the sidecar.
const FORWARDED_REQUEST_HEADERS: [HeaderName; 3] =
    [header::CONTENT_TYPE, header::ACCEPT, header::AUTHORIZATION];

/// `POST /v1/patch_sweep` — reverse-proxy to the patch sidecar (or 501).
pub async fn patch_sweep(State(state): State<Arc<AppState>>, req: Request) -> Response {
    proxy(state, req, "/v1/patch_sweep".to_string()).await
}

/// `DELETE /v1/patch_source/{run_id}` — reverse-proxy to the patch sidecar
/// (or 501).
pub async fn drop_patch_source(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
    req: Request,
) -> Response {
    proxy(state, req, format!("/v1/patch_source/{run_id}")).await
}

/// Forward one request to the configured sidecar, streaming the response body
/// through incrementally. Returns 501 when no sidecar is configured.
async fn proxy(state: Arc<AppState>, req: Request, upstream_path: String) -> Response {
    let Some(sidecar) = state.patch_sidecar() else {
        return sidecar_unavailable();
    };

    let (parts, body) = req.into_parts();
    let method = parts.method;
    let headers = parts.headers;

    // Buffer the (small) request body before forwarding. Only the *response*
    // must stream; the request grid is bounded JSON.
    let body_bytes = match axum::body::to_bytes(body, MAX_REQUEST_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(err) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("failed to read request body: {err}"),
            );
        }
    };

    match forward(sidecar, method, &upstream_path, &headers, body_bytes).await {
        Ok(response) => response,
        Err(err) => error_response(
            StatusCode::BAD_GATEWAY,
            "server_error",
            format!("patch sidecar request failed: {err}"),
        ),
    }
}

/// Send the request upstream and build a streaming passthrough response.
async fn forward(
    sidecar: &PatchSidecar,
    method: Method,
    upstream_path: &str,
    headers: &HeaderMap,
    body: axum::body::Bytes,
) -> Result<Response, reqwest::Error> {
    let url = sidecar.url(upstream_path);
    let mut builder = sidecar.client().request(method, url);
    for name in &FORWARDED_REQUEST_HEADERS {
        if let Some(value) = headers.get(name) {
            builder = builder.header(name, value);
        }
    }

    let upstream = builder.body(body).send().await?;

    let status = upstream.status();
    let mut response = Response::builder().status(status);
    for (name, value) in upstream.headers() {
        // Drop hop-by-hop / framing headers: the body is re-chunked by axum, so
        // an upstream Content-Length / Transfer-Encoding would be wrong.
        if is_hop_by_hop(name) {
            continue;
        }
        response = response.header(name, value);
    }

    // Stream the body through. Dropping this stream (client disconnect) drops
    // the upstream connection, so the Python side cancels the in-flight sweep.
    let stream = upstream.bytes_stream();
    Ok(response
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response()))
}

/// Whether a response header is hop-by-hop and must not be forwarded.
fn is_hop_by_hop(name: &HeaderName) -> bool {
    name == header::CONTENT_LENGTH
        || name == header::TRANSFER_ENCODING
        || name == header::CONNECTION
}

/// 501 response returned when no patch sidecar is configured.
fn sidecar_unavailable() -> Response {
    error_response(
        StatusCode::NOT_IMPLEMENTED,
        "server_error",
        "activation-patching sweeps require the internal Python patch sidecar, \
         which is not running. Launch vLLM with the Rust frontend and \
         --enable-patching (and VLLM_RUST_PATCH_SIDECAR unset or 1), or use the \
         Python api_server directly. See the \"Activation patching support\" \
         section of rust/README.md."
            .to_string(),
    )
}

/// Build an OpenAI-style JSON error response with the given status.
fn error_response(status: StatusCode, error_type: &str, message: String) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": error_type,
                "param": null,
                "code": error_type,
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests;
