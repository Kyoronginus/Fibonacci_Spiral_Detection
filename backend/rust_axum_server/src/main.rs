use axum::{
    extract::DefaultBodyLimit,
    http::{header, Method, StatusCode, HeaderValue},
    response::{Response, IntoResponse},
    routing::post,
    Router,
    Json,
};
use axum_typed_multipart::{FieldData, TryFromMultipart, TypedMultipart};
use bytes::Bytes;
use std::env;
use std::io::Cursor;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use image::io::Reader as ImageReader;
use std::cmp::min;
use base64::{Engine as _, engine::general_purpose};
use linfa::traits::Fit;

mod preprocessing;
mod clustering;
mod spiral_fit;
mod visualization;

use preprocessing::smart_resize;
use clustering::{extract_object_centroids, find_optimal_k};
use spiral_fit::{optimize_spiral_with_golden_ratio, calculate_composition_score, SpiralParams};
use visualization::draw_result;

#[derive(TryFromMultipart)]
struct AnalyzeRequest {
    #[form_data(field_name = "file", limit = "25MiB")]
    file: FieldData<Bytes>,
    k: String,
    b_weight: String,
}

#[derive(TryFromMultipart)]
struct PreviewRequest {
    #[form_data(field_name = "file", limit = "25MiB")]
    file: FieldData<Bytes>,
    k: String,
}

#[derive(serde::Serialize)]
struct AnalyzeResponse {
    score: f64,
    b_value: f64,
    golden_b: f64,
    image_base64: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "rust_axum_server=debug".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cors = CorsLayer::permissive();

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

async fn analyze_handler(
    TypedMultipart(payload): TypedMultipart<AnalyzeRequest>,
) -> Result<Json<AnalyzeResponse>, (StatusCode, String)> {
    let file_data = payload.file.contents;
    let k_val: usize = payload.k.parse().unwrap_or(0);
    let b_weight_val: f64 = payload.b_weight.parse().unwrap_or(20000.0);

    let img = ImageReader::new(Cursor::new(file_data))
        .with_guessed_format()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to read image: {}", e)))?
        .decode()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to decode image: {}", e)))?;

    let resized_image = smart_resize(&img, 2048);
    let (initial_centroids, _) = extract_object_centroids(&resized_image)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to extract centroids: {}", e)))?;

    if initial_centroids.len() < 3 {
        return Err((StatusCode::BAD_REQUEST, "分析対象オブジェクトが3つ未満です。".to_string()));
    }

    let optimal_k = if k_val > 0 {
        min(k_val, initial_centroids.len())
    } else {
        find_optimal_k(&initial_centroids, 10)
    };

    let final_k = if optimal_k < 2 && initial_centroids.len() >= 2 {
        2
    } else {
        optimal_k
    };

    if initial_centroids.len() < 2 {
        return Err((StatusCode::BAD_REQUEST, "分析対象オブジェクトが2つ未満です。".to_string()));
    }

    // Perform KMeans clustering
    let records = ndarray::Array2::from_shape_vec((initial_centroids.len(), 2), initial_centroids.iter().flat_map(|p| p.to_vec()).collect()).unwrap();
    let dataset = linfa::DatasetBase::new(records, ());
    
    let model = linfa_clustering::KMeans::params(final_k)
        .max_n_iterations(100)
        .tolerance(1e-5)
        .fit(&dataset)
        .expect("KMeans failed");
    
    let clustered_centroids_array = model.centroids();
    let mut clustered_centroids = Vec::new();
    for row in clustered_centroids_array.outer_iter() {
        clustered_centroids.push([row[0], row[1]]);
    }

    let image_shape = (resized_image.width(), resized_image.height());
    let best_params = optimize_spiral_with_golden_ratio(&clustered_centroids, image_shape, b_weight_val);

    let distance_score = calculate_composition_score(&best_params, &clustered_centroids, image_shape);
    let score_fit = (-0.05 * distance_score).exp();
    let golden_b = 0.30635;
    let score_golden = (-50.0 * (best_params.b - golden_b).abs()).exp();
    let final_score = (0.6 * score_fit + 0.4 * score_golden) * 100.0;

    let result_image = draw_result(&resized_image, &initial_centroids, &clustered_centroids, Some(&best_params))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to draw result: {}", e)))?;

    let mut buffer = Cursor::new(Vec::new());
    result_image.write_to(&mut buffer, image::ImageOutputFormat::Png)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to encode image: {}", e)))?;
    
    let image_base64 = general_purpose::STANDARD.encode(buffer.get_ref());

    Ok(Json(AnalyzeResponse {
        score: (final_score * 10.0).round() / 10.0,
        b_value: (best_params.b * 10000.0).round() / 10000.0,
        golden_b,
        image_base64: format!("data:image/png;base64,{}", image_base64),
    }))
}

async fn preview_handler(
    TypedMultipart(payload): TypedMultipart<PreviewRequest>,
) -> Result<Response, (StatusCode, String)> {
    let file_data = payload.file.contents;
    let k_val: usize = payload.k.parse().unwrap_or(0);

    let img = ImageReader::new(Cursor::new(file_data))
        .with_guessed_format()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to read image: {}", e)))?
        .decode()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to decode image: {}", e)))?;

    let resized_image = smart_resize(&img, 2048);
    let (initial_centroids, _) = extract_object_centroids(&resized_image)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to extract centroids: {}", e)))?;

    let clustered_centroids = if initial_centroids.len() < 2 {
        initial_centroids.clone()
    } else if k_val == 1 {
        let sum_x: f64 = initial_centroids.iter().map(|p| p[0]).sum();
        let sum_y: f64 = initial_centroids.iter().map(|p| p[1]).sum();
        vec![[sum_x / initial_centroids.len() as f64, sum_y / initial_centroids.len() as f64]]
    } else if k_val >= 2 {
        let k_to_use = min(k_val, initial_centroids.len());
        let records = ndarray::Array2::from_shape_vec((initial_centroids.len(), 2), initial_centroids.iter().flat_map(|p| p.to_vec()).collect()).unwrap();
        let dataset = linfa::DatasetBase::new(records, ());
        let model = linfa_clustering::KMeans::params(k_to_use)
            .max_n_iterations(100)
            .tolerance(1e-5)
            .fit(&dataset)
            .expect("KMeans failed");
        
        let centroids_array = model.centroids();
        let mut centroids = Vec::new();
        for row in centroids_array.outer_iter() {
            centroids.push([row[0], row[1]]);
        }
        centroids
    } else {
        Vec::new()
    };

    let preview_image = draw_result(&resized_image, &initial_centroids, &clustered_centroids, None)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to draw result: {}", e)))?;

    let mut buffer = Cursor::new(Vec::new());
    preview_image.write_to(&mut buffer, image::ImageOutputFormat::Png)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to encode image: {}", e)))?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/png")
        .body(axum::body::Body::from(buffer.into_inner()))
        .unwrap())
}
