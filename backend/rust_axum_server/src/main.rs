use axum::{
    extract::DefaultBodyLimit,
    routing::post,
    Router,
};
use std::env;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
};
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod preprocessing;
mod clustering;
mod spiral_fit;
mod visualization;
mod handlers; 

use handlers::{analyze_handler, preview_handler};

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "rust_axum_server=debug".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cors = CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    let app = Router::new()
        .route("/analyze/", post(analyze_handler))
        .route("/preview_clusters/", post(preview_handler))
        .layer(TraceLayer::new_for_http())
        .layer(DefaultBodyLimit::max(25 * 1024 * 1024))
        .layer(cors);

    let port = env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let addr = format!("0.0.0.0:{}", port);
    tracing::debug!("Preparing to listen on {}", addr);

    let listener = TcpListener::bind(&addr).await.unwrap();
    tracing::debug!("Successfully listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}
