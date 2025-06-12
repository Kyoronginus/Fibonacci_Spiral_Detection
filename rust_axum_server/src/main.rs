use axum::{
    extract::DefaultBodyLimit,
    http::{header, Method, StatusCode, HeaderValue},
    response::Response,
    routing::post,
    Router,
};
use axum_typed_multipart::{FieldData, TryFromMultipart, TypedMultipart};
use bytes::Bytes;
use std::env; // ★★★ 環境変数を読み込むために追加 ★★★
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(TryFromMultipart)]
struct UploadRequest {
    #[form_data(field_name = "file")]
    file: FieldData<Bytes>,
    k: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry().with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "rust_axum_server=debug".into())).with(tracing_subscriber::fmt::layer()).init();
    
    // CORS設定
    let frontend_url = env::var("FRONTEND_URL").unwrap_or_else(|_| "http://localhost:5173".to_string());
    let cors = CorsLayer::new()
        .allow_origin(frontend_url.parse::<HeaderValue>().unwrap())
        .allow_methods([Method::POST])
        .allow_headers(Any);

    let app = Router::new()
        .route("/upload", post(upload_handler))
        .layer(TraceLayer::new_for_http())
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .layer(cors);

    // ★★★ ポート番号を環境変数から取得するように修正 ★★★
    // `PORT`環境変数が設定されていればそれを使い、なければ`3000`をデフォルト値とする
    let port = env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let addr = format!("0.0.0.0:{}", port);

    tracing::debug!("Preparing to listen on {}", addr);
    
    let listener = TcpListener::bind(&addr).await.unwrap();
    tracing::debug!("Successfully listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

// upload_handler関数は変更なし
async fn upload_handler(
    TypedMultipart(payload): TypedMultipart<UploadRequest>,
) -> Result<Response, (StatusCode, String)> {
    let file_name = payload.file.metadata.file_name.unwrap_or("unknown.dat".to_string());
    let file_data = payload.file.contents;
    let k_value = payload.k;

    tracing::debug!("Received file '{}' ({} bytes) with k='{}'", file_name, file_data.len(), k_value);

    let python_service_url = env::var("PYTHON_SERVICE_URL")
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "PYTHON_SERVICE_URL env var not set".to_string()))?;
        
    let client = reqwest::Client::new();
    let part = reqwest::multipart::Part::bytes(file_data.to_vec()).file_name(file_name);
    let k_part = reqwest::multipart::Part::text(k_value);
    let form = reqwest::multipart::Form::new().part("file", part).part("k", k_part);

    tracing::debug!("Forwarding to Python service at {}", python_service_url);
    let res = client.post(&python_service_url).multipart(form).send().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Python service request failed: {}", e)))?;
    
    if res.status().is_success() {
        let image_bytes = res.bytes().await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to read image from Python service: {}", e)))?;
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "image/png")
            .body(image_bytes.into())
            .unwrap())
    } else {
        let error_text = res.text().await.unwrap_or_else(|_| "Unknown error from Python service".to_string());
        tracing::error!("Python service returned an error: {}", error_text);
        Err((StatusCode::BAD_GATEWAY, format!("Python service error: {}", error_text)))
    }
}