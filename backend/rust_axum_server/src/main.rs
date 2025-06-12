use axum::{
    extract::DefaultBodyLimit,
    http::{header, Method, StatusCode, HeaderValue},
    response::Response,
    routing::post,
    Router,
};
use axum_typed_multipart::{FieldData, TryFromMultipart, TypedMultipart};
use bytes::Bytes;
use std::env;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

// ★★★ UploadRequest構造体に `b_weight` を追加 ★★★
#[derive(TryFromMultipart)]
struct UploadRequest {
    #[form_data(field_name = "file", limit = "25MiB")]
    file: FieldData<Bytes>,
    k: String,
    b_weight: String,
}

#[tokio::main]
async fn main() {
    // (main関数の中身は変更なし)
    tracing_subscriber::registry().with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "rust_axum_server=debug".into())).with(tracing_subscriber::fmt::layer()).init();
    let frontend_url = env::var("FRONTEND_URL").unwrap_or_else(|_| "http://localhost:5173".to_string());
    let cors = CorsLayer::new().allow_origin(frontend_url.parse::<HeaderValue>().unwrap()).allow_methods([Method::POST]).allow_headers(Any);
    let app = Router::new().route("/upload", post(upload_handler)).layer(TraceLayer::new_for_http()).layer(DefaultBodyLimit::max(25 * 1024 * 1024)).layer(cors);
    let port = env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let addr = format!("0.0.0.0:{}", port);
    tracing::debug!("Preparing to listen on {}", addr);
    let listener = TcpListener::bind(&addr).await.unwrap();
    tracing::debug!("Successfully listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

// ★★★ `b_weight`もPythonに転送するように修正 ★★★
async fn upload_handler(
    TypedMultipart(payload): TypedMultipart<UploadRequest>,
) -> Result<Response, (StatusCode, String)> {
    let file_name = payload.file.metadata.file_name.unwrap_or("unknown.dat".to_string());
    let file_data = payload.file.contents;
    let k_value = payload.k;
    let b_weight_value = payload.b_weight; // b_weightの値を取得

    tracing::debug!("Received file '{}' ({} bytes) with k='{}' and b_weight='{}'", file_name, file_data.len(), k_value, b_weight_value);

    let python_service_url = env::var("PYTHON_SERVICE_URL")
        .unwrap_or_else(|_| "https://python-analysis-server-179718527697.asia-northeast1.run.app".to_string());
        
    let client = reqwest::Client::new();
    
    // Pythonに送るためのマルチパートフォームを構築
    let file_part = reqwest::multipart::Part::bytes(file_data.to_vec()).file_name(file_name);
    let k_part = reqwest::multipart::Part::text(k_value);
    let b_weight_part = reqwest::multipart::Part::text(b_weight_value); // b_weightもパートとして作成

    let form = reqwest::multipart::Form::new()
        .part("file", file_part)
        .part("k", k_part)
        .part("b_weight", b_weight_part); // ★★★ b_weightパートをフォームに追加 ★★★

    tracing::debug!("Forwarding to Python service at {}", python_service_url);
    let request = client.post(&python_service_url)
        .multipart(form);

    // If the Python service requires authentication, add the necessary headers
    // For Google Cloud Run, you might need to add an Authorization header
    // This depends on your specific setup
    let res = request.send().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Python service request failed: {}", e)))?;
    
    if res.status().is_success() {
        // ... (以降の成功・失敗処理は変更なし) ...
        let json_response = res.bytes().await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to read json from Python service: {}", e)))?;
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .body(json_response.into())
            .unwrap())
    } else {
        let error_text = res.text().await.unwrap_or_else(|_| "Unknown error from Python service".to_string());
        tracing::error!("Python service returned an error: {}", error_text);
        Err((StatusCode::BAD_GATEWAY, format!("Python service error: {}", error_text)))
    }
}


