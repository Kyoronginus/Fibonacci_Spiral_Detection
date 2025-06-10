use axum::{
    extract::{DefaultBodyLimit, Multipart},
    http::{header, Method, StatusCode},
    response::Response,
    routing::post,
    Router,
};
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    // (main関数は変更なし)
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rust_axum_server=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::POST])
        .allow_headers(Any);

    let app = Router::new()
        .route("/upload", post(upload_handler))
        .layer(TraceLayer::new_for_http())
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .layer(cors);

    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();
    tracing::debug!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

// ★★★ 最終修正版のupload_handler ★★★
async fn upload_handler(mut multipart: Multipart) -> Result<Response, (StatusCode, String)> {
    while let Some(mut field) = multipart.next_field().await.map_err(|err| {
        (StatusCode::BAD_REQUEST, format!("Invalid multipart form: {}", err))
    })? {
        if let Some(field_name) = field.name() {
            if field_name == "file" {
                if let Some(file_name) = field.file_name() {
                    let file_name = file_name.to_string();
                    tracing::debug!("Processing file field: '{}'", file_name);

                    // ★★★ データをチャンク単位で読み込むように変更 ★★★
                    let mut data_vec: Vec<u8> = Vec::new();
                    while let Some(chunk) = field.chunk().await.map_err(|err| {
                        (StatusCode::BAD_REQUEST, format!("Error reading chunk: {}", err))
                    })? {
                        data_vec.extend_from_slice(&chunk);
                    }
                    tracing::debug!("Received file '{}' with size: {} bytes", file_name, data_vec.len());

                    // --- Pythonサービス呼び出し ---
                    let python_service_url = "http://127.0.0.1:8000/analyze/";
                    let client = reqwest::Client::new();
                    // data_vecは既にVec<u8>なので、.to_vec()は不要
                    let part = reqwest::multipart::Part::bytes(data_vec).file_name(file_name);
                    let form = reqwest::multipart::Form::new().part("file", part);

                    tracing::debug!("Forwarding to Python service...");
                    let res = client.post(python_service_url).multipart(form).send().await
                        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Python service request failed: {}", e)))?;
                    
                    if res.status().is_success() {
                        let image_bytes = res.bytes().await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to read image from Python service: {}", e)))?;
                        return Ok(Response::builder()
                            .status(StatusCode::OK)
                            .header(header::CONTENT_TYPE, "image/png")
                            .body(image_bytes.into())
                            .unwrap());
                    } else {
                        let error_text = res.text().await.unwrap_or_else(|_| "Unknown error from Python service".to_string());
                        tracing::error!("Python service returned an error: {}", error_text);
                        return Err((StatusCode::BAD_GATEWAY, format!("Python service error: {}", error_text)));
                    }
                }
            }
        }
    }
    
    Err((StatusCode::BAD_REQUEST, "Field 'file' not found in upload".to_string()))
}