use axum::{
    http::{header, StatusCode},
    response::{Response, IntoResponse},
    Json,
};
use axum_typed_multipart::{FieldData, TryFromMultipart, TypedMultipart};
use bytes::Bytes;
use std::io::Cursor;
use image::io::Reader as ImageReader;
use image::DynamicImage;
use std::cmp::min;
use base64::{Engine as _, engine::general_purpose};

use crate::preprocessing::smart_resize;
use crate::clustering::{extract_object_centroids, find_optimal_k, perform_clustering};
use crate::spiral_fit::{optimize_spiral_with_golden_ratio, calculate_composition_score};
use crate::visualization::draw_result;

#[derive(TryFromMultipart)]
pub struct AnalyzeRequest {
    #[form_data(field_name = "file", limit = "25MiB")]
    pub file: FieldData<Bytes>,
    pub k: String,
    pub b_weight: String,
}

#[derive(TryFromMultipart)]
pub struct PreviewRequest {
    #[form_data(field_name = "file", limit = "25MiB")]
    pub file: FieldData<Bytes>,
    pub k: String,
}

#[derive(serde::Serialize)]
pub struct AnalyzeResponse {
    pub score: f64,
    pub b_value: f64,
    pub golden_b: f64,
    pub image_base64: String,
}

fn load_and_resize_image(file_data: Bytes) -> Result<DynamicImage, (StatusCode, String)> {
    let img = ImageReader::new(Cursor::new(file_data))
        .with_guessed_format()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to read image: {}", e)))?
        .decode()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to decode image: {}", e)))?;

    Ok(smart_resize(&img, 1024))
}

pub async fn analyze_handler(
    TypedMultipart(payload): TypedMultipart<AnalyzeRequest>,
) -> Result<Json<AnalyzeResponse>, (StatusCode, String)> {
    let k_val: usize = payload.k.parse().unwrap_or(0);
    let b_weight_val: f64 = payload.b_weight.parse().unwrap_or(20000.0);

    let resized_image = load_and_resize_image(payload.file.contents)?;
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

    let clustered_centroids = perform_clustering(&initial_centroids, final_k)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Clustering failed: {}", e)))?;

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

pub async fn preview_handler(
    TypedMultipart(payload): TypedMultipart<PreviewRequest>,
) -> Result<Response, (StatusCode, String)> {
    let k_val: usize = payload.k.parse().unwrap_or(0);

    let resized_image = load_and_resize_image(payload.file.contents)?;
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
        perform_clustering(&initial_centroids, k_to_use)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Clustering failed: {}", e)))?
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
