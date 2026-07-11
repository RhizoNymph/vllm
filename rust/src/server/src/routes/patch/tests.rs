//! Tests for the patch-sweep reverse proxy, backed by a mock upstream axum
//! server standing in for the Python patch sidecar.

use std::time::{Duration, Instant};

use axum::Router;
use axum::body::{Body, Bytes, to_bytes};
use axum::extract::Path;
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::{delete, post};
use futures::StreamExt as _;

use super::*;
use crate::state::PatchSidecar;

/// Spawn `app` on an ephemeral loopback port and return its base URL.
async fn spawn_mock_sidecar(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind mock sidecar");
    let addr = listener.local_addr().expect("mock sidecar addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve mock sidecar");
    });
    format!("http://{addr}")
}

fn json_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
    headers
}

#[tokio::test]
async fn proxy_forwards_post_body_and_returns_json() {
    // Mock echoes the received body back as JSON so the test can assert the
    // request body reached the upstream unchanged.
    async fn echo(body: Bytes) -> impl IntoResponse {
        ([(header::CONTENT_TYPE, "application/json")], body)
    }
    let base_url = spawn_mock_sidecar(Router::new().route("/v1/patch_sweep", post(echo))).await;
    let sidecar = PatchSidecar::new_for_test(base_url);

    let request_body = Bytes::from_static(br#"{"layers":[1,2],"positions":[3]}"#);
    let response = forward(
        &sidecar,
        Method::POST,
        "/v1/patch_sweep",
        &json_headers(),
        request_body.clone(),
    )
    .await
    .expect("forward");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/json"
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(body, request_body);
}

#[tokio::test]
async fn proxy_streams_sse_chunks_incrementally() {
    // Mock emits one SSE chunk, waits, then emits a second: a buffering proxy
    // would deliver both at once, so an observable gap proves passthrough.
    async fn sse() -> impl IntoResponse {
        let stream = futures::stream::unfold(0u8, |state| async move {
            match state {
                0 => Some((
                    Ok::<_, std::io::Error>(Bytes::from_static(b"data: first\n\n")),
                    1,
                )),
                1 => {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    Some((Ok(Bytes::from_static(b"data: second\n\n")), 2))
                }
                _ => None,
            }
        });
        (
            [(header::CONTENT_TYPE, "text/event-stream")],
            Body::from_stream(stream),
        )
    }
    let base_url = spawn_mock_sidecar(Router::new().route("/v1/patch_sweep", post(sse))).await;
    let sidecar = PatchSidecar::new_for_test(base_url);

    let response = forward(
        &sidecar,
        Method::POST,
        "/v1/patch_sweep",
        &json_headers(),
        Bytes::from_static(b"{}"),
    )
    .await
    .expect("forward");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/event-stream"
    );

    let start = Instant::now();
    let mut stream = response.into_body().into_data_stream();

    let first = stream.next().await.expect("first chunk").expect("chunk ok");
    let first_at = start.elapsed();
    assert_eq!(&first[..], b"data: first\n\n");

    let second = stream.next().await.expect("second chunk").expect("chunk ok");
    let second_at = start.elapsed();
    assert_eq!(&second[..], b"data: second\n\n");

    // The second chunk must arrive meaningfully after the first (not buffered).
    assert!(
        second_at - first_at >= Duration::from_millis(100),
        "expected an incremental gap between SSE chunks, got {:?} then {:?}",
        first_at,
        second_at,
    );
}

#[tokio::test]
async fn proxy_forwards_delete_with_path() {
    async fn drop_run(Path(run_id): Path<String>) -> impl IntoResponse {
        (
            [(header::CONTENT_TYPE, "application/json")],
            format!(r#"{{"dropped":true,"run_id":"{run_id}"}}"#),
        )
    }
    let base_url =
        spawn_mock_sidecar(Router::new().route("/v1/patch_source/{run_id}", delete(drop_run)))
            .await;
    let sidecar = PatchSidecar::new_for_test(base_url);

    let response = forward(
        &sidecar,
        Method::DELETE,
        "/v1/patch_source/clean-run-7",
        &HeaderMap::new(),
        Bytes::new(),
    )
    .await
    .expect("forward");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("clean-run-7"), "unexpected body: {text}");
    assert!(text.contains("\"dropped\":true"), "unexpected body: {text}");
}

#[tokio::test]
async fn unconfigured_sidecar_returns_501() {
    let response = sidecar_unavailable();
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("patch sidecar"), "unexpected body: {text}");
    assert!(text.contains("README.md"), "unexpected body: {text}");
    assert!(
        text.contains("\"type\":\"server_error\""),
        "unexpected body: {text}"
    );
}
