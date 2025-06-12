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
    tracing_subscriber::registry().with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "rust_axum_server=debug,tower_http=debug".into())).with(tracing_subscriber::fmt::layer()).init();
    let cors = CorsLayer::new().allow_origin(Any).allow_methods([Method::POST]).allow_headers(Any);
    let app = Router::new().route("/upload", post(upload_handler)).layer(TraceLayer::new_for_http()).layer(DefaultBodyLimit::max(10 * 1024 * 1024)).layer(cors);
    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();
    tracing::debug!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

async fn upload_handler(mut multipart: Multipart) -> Result<Response, (StatusCode, String)> {
    let mut file_data = None;
    let mut k_value = None;
    let mut b_weight = None; // ★★★ 追加 ★★★

    // --- ステップ1: まずリクエストの全フィールドを読み込む ---
    while let Some(field) = multipart.next_field().await.map_err(|err| {
        (StatusCode::BAD_REQUEST, format!("Invalid multipart form: {}", err))
    })? {
        let name = if let Some(name) = field.name() {
            name.to_string()
        } else {
            continue;
        };

        if name == "file" {
            let file_name = field.file_name().unwrap_or("unknown.dat").to_string();
            let data = field.bytes().await.map_err(|err| {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to read file bytes: {}", err))
            })?;
            file_data = Some((file_name, data));
        } else if name == "k" {
            let data = field.text().await.map_err(|err| {
                (StatusCode::BAD_REQUEST, format!("Failed to read 'k' value: {}", err))
            })?;
            k_value = Some(data);
        } else if name == "b_weight" {
            let data = field.text().await.map_err(|err| {
                (StatusCode::BAD_REQUEST, format!("Failed to read 'b_weight' value: {}", err))
            })?;
            b_weight = Some(data);
        }
    }

    // --- ステップ2: 必要なデータが全て揃っているか確認 ---
    if let (Some((file_name, file_data)), Some(k_value)) = (file_data, k_value) {
        // b_weightがなければデフォルト値
        let b_weight_value = b_weight.unwrap_or_else(|| "100".to_string());

        tracing::debug!("Received file '{}' ({} bytes) with k='{}', b_weight='{}'", file_name, file_data.len(), k_value, b_weight_value);

        // --- ステップ3: Pythonサービスを呼び出す (データが揃ってから) ---
        let python_service_url = "http://127.0.0.1:8000/analyze/";
        let client = reqwest::Client::new();

        let file_part = reqwest::multipart::Part::bytes(file_data.to_vec()).file_name(file_name);
        let k_part = reqwest::multipart::Part::text(k_value);
        let b_weight_part = reqwest::multipart::Part::text(b_weight_value);

        let form = reqwest::multipart::Form::new()
            .part("file", file_part)
            .part("k", k_part)
            .part("b_weight", b_weight_part); // ★★★ 追加 ★★★

        tracing::debug!("Forwarding to Python service...");
        let res = client.post(python_service_url).multipart(form).send().await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Python service request failed: {}", e)))?;

        if res.status().is_success() {
            let json_bytes = res.bytes().await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to read response from Python service: {}", e)))?;
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/json")
                .body(json_bytes.into())
                .unwrap())
        } else {
            let error_text = res.text().await.unwrap_or_else(|_| "Unknown error from Python service".to_string());
            tracing::error!("Python service returned an error: {}", error_text);
            Err((StatusCode::BAD_GATEWAY, format!("Python service error: {}", error_text)))
        }
    } else {
        Err((StatusCode::BAD_REQUEST, "Request must include both 'file' and 'k' fields.".to_string()))
    }
}