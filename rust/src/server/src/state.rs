use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use tokio::sync::Mutex;
use tokio::time::{Duration, Instant, sleep_until};
use tracing::warn;
use vllm_chat::ChatLlm;
use vllm_engine_core_client::EngineCoreClient;

const SHUTDOWN_REFCOUNT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Shared router state for the minimal single-model OpenAI server.
pub struct AppState {
    /// All public model IDs served by this frontend. The first entry is the
    /// primary ID used in responses; all entries are valid in requests.
    served_model_names: Vec<String>,
    /// Shared chat facade used by all requests.
    pub chat: ChatLlm,
    /// Whether to log a summary line for each completed request.
    pub enable_log_requests: bool,
    /// Names of steering modules currently registered with the engine workers.
    /// Seeded at startup and mutated by the runtime steering endpoints; requests
    /// referencing an unknown `steering_name` are rejected up front.
    steering_module_names: RwLock<HashSet<String>>,
    /// Serializes runtime steering-registry mutations so concurrent
    /// register/unregister requests cannot interleave their broadcasts.
    steering_mutation_lock: Mutex<()>,
    /// Number of in-flight inference requests currently owned by this frontend.
    server_load: AtomicU64,
}

impl AppState {
    /// Construct one application state instance.
    ///
    /// `served_model_names` must be non-empty; the first entry is the primary
    /// model ID returned in API responses.
    ///
    /// # Panics
    ///
    /// Panics if `served_model_names` is empty.
    pub fn new(served_model_names: Vec<String>, chat: ChatLlm) -> Self {
        assert!(
            !served_model_names.is_empty(),
            "served_model_names must not be empty"
        );
        Self {
            served_model_names,
            chat,
            enable_log_requests: false,
            steering_module_names: RwLock::new(HashSet::new()),
            steering_mutation_lock: Mutex::new(()),
            server_load: AtomicU64::new(0),
        }
    }

    /// Enable per-request completion logging.
    pub fn with_log_requests(mut self, enabled: bool) -> Self {
        self.enable_log_requests = enabled;
        self
    }

    /// Seed the set of steering modules registered with the engine workers.
    pub fn with_steering_module_names(mut self, names: HashSet<String>) -> Self {
        self.steering_module_names = RwLock::new(names);
        self
    }

    /// Validate a request's `steering_name` against the registered modules.
    ///
    /// Returns a human-readable error message when the name is present but not
    /// registered (listing the available modules), or `None` when the request
    /// omits `steering_name` or references a known module.
    pub fn steering_module_error(&self, steering_name: Option<&str>) -> Option<String> {
        let name = steering_name?;
        let registered = self.steering_module_names.read().expect("registry lock");
        if registered.contains(name) {
            return None;
        }
        let mut available: Vec<&str> = registered.iter().map(String::as_str).collect();
        available.sort_unstable();
        Some(format!(
            "Unknown steering module '{name}'. Available: [{}]",
            available.join(", ")
        ))
    }

    /// Return the sorted names of currently registered steering modules.
    pub fn list_steering_modules(&self) -> Vec<String> {
        let registered = self.steering_module_names.read().expect("registry lock");
        let mut names: Vec<String> = registered.iter().cloned().collect();
        names.sort_unstable();
        names
    }

    /// Whether a steering module is currently registered.
    pub fn is_steering_module_registered(&self, name: &str) -> bool {
        self.steering_module_names.read().expect("registry lock").contains(name)
    }

    /// Replace the entire registered-name set (after a `replace=true` register).
    pub fn set_steering_module_names(&self, names: HashSet<String>) {
        *self.steering_module_names.write().expect("registry lock") = names;
    }

    /// Add names to the registered set (after a `replace=false` register).
    pub fn extend_steering_module_names(&self, names: impl IntoIterator<Item = String>) {
        self.steering_module_names.write().expect("registry lock").extend(names);
    }

    /// Remove one name from the registered set, returning whether it was present.
    pub fn remove_steering_module_name(&self, name: &str) -> bool {
        self.steering_module_names.write().expect("registry lock").remove(name)
    }

    /// Acquire the registry-mutation lock, serializing runtime register and
    /// unregister operations against each other.
    pub async fn lock_steering_mutations(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.steering_mutation_lock.lock().await
    }

    /// The primary model name echoed back in API responses (the first served
    /// name).
    pub fn primary_model_name(&self) -> &str {
        self.served_model_names.first().map(String::as_str).unwrap_or_default()
    }

    /// All model names served by this frontend.
    pub fn served_model_names(&self) -> &[String] {
        &self.served_model_names
    }

    /// Return a reference to the underlying engine core client for utility
    /// calls.
    pub(crate) fn engine_core_client(&self) -> &EngineCoreClient {
        self.chat.engine_core_client()
    }

    /// Return the current in-flight inference request count for the `/load`
    /// endpoint.
    pub fn server_load(&self) -> u64 {
        self.server_load.load(Ordering::Relaxed)
    }

    /// Increment the in-flight inference request count, called by the load
    /// tracking middleware.
    pub(crate) fn increment_server_load(&self) {
        self.server_load.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the in-flight inference request count, called by the load
    /// tracking middleware.
    pub(crate) fn decrement_server_load(&self) {
        self.server_load.fetch_sub(1, Ordering::Relaxed);
    }

    /// Wait until all request-owned references are dropped, then shut down the
    /// engine client.
    ///
    /// If the deadline elapses while request/connection tasks still hold state
    /// references, skip the clean engine-client shutdown and let process
    /// teardown reclaim the remaining resources.
    pub async fn shutdown(mut self: Arc<Self>, deadline: Instant) -> anyhow::Result<()> {
        loop {
            match Arc::try_unwrap(self) {
                Ok(state) => {
                    state.chat.shutdown().await?;
                    return Ok(());
                }
                Err(state) => self = state,
            }
            let ref_count = Arc::strong_count(&self);

            let now = Instant::now();
            if now >= deadline {
                warn!(
                    ref_count,
                    "shutdown deadline elapsed before app state became idle; skipping engine-client shutdown"
                );
                return Ok(());
            }

            sleep_until(std::cmp::min(
                deadline,
                now + SHUTDOWN_REFCOUNT_POLL_INTERVAL,
            ))
            .await;
        }
    }
}
