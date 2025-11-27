use image::{DynamicImage, GenericImageView, GrayImage, Luma, imageops};
use imageproc::contours::find_contours;
use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use rand::thread_rng;
use anyhow::Result;

pub fn extract_object_centroids(image: &DynamicImage) -> Result<(Vec<[f64; 2]>, Vec<Vec<imageproc::point::Point<i32>>>)> {
    let gray = image.to_luma8();

    // OpenCV's adaptiveThreshold with ADAPTIVE_THRESH_GAUSSIAN_C
    // blockSize = 11, C = 2.0
    // Gaussian blur with sigma roughly related to blockSize.
    // OpenCV uses sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 if not specified.
    // For ksize=11, sigma ≈ 0.3*(5-1) + 0.8 = 2.0? No, let's check.
    // Actually, we can just use a sigma that gives a similar effect.
    // Let's use sigma = 2.0 as a good approximation for block size 11.
    
    let blurred = imageops::blur(&gray, 2.0);

    let width = gray.width();
    let height = gray.height();
    let mut thresh_image = GrayImage::new(width, height);
    let c_val = 2.0;

    for y in 0..height {
        for x in 0..width {
            let pixel_val = gray.get_pixel(x, y)[0] as f32;
            let blurred_val = blurred.get_pixel(x, y)[0] as f32;
            
            // THRESH_BINARY_INV:
            // if src(x,y) > T(x,y) { 0 } else { 255 }
            // T(x,y) = weighted_sum(x,y) - C
            
            let threshold = blurred_val - c_val;
            
            if pixel_val > threshold {
                thresh_image.put_pixel(x, y, Luma([0]));
            } else {
                thresh_image.put_pixel(x, y, Luma([255]));
            }
        }
    }

    // Find contours
    // imageproc find_contours returns Vec<Contour<i32>>
    // We need to filter by area.
    let contours = find_contours(&thresh_image);
    
    let mut centroids = Vec::new();
    let mut filtered_contours_points = Vec::new();

    for contour in contours {
        // Calculate area
        let area = polygon_area(&contour.points);
        
        if area.abs() > 50.0 {
            if let Some(centroid) = polygon_centroid(&contour.points) {
                centroids.push(centroid);
                filtered_contours_points.push(contour.points);
            }
        }
    }

    Ok((centroids, filtered_contours_points))
}

fn polygon_area(points: &[imageproc::point::Point<i32>]) -> f64 {
    let n = points.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += (points[i].x as f64 * points[j].y as f64) - (points[j].x as f64 * points[i].y as f64);
    }
    0.5 * area
}

fn polygon_centroid(points: &[imageproc::point::Point<i32>]) -> Option<[f64; 2]> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut signed_area = 0.0;
    
    for i in 0..n {
        let j = (i + 1) % n;
        let a = (points[i].x as f64 * points[j].y as f64) - (points[j].x as f64 * points[i].y as f64);
        signed_area += a;
        cx += (points[i].x as f64 + points[j].x as f64) * a;
        cy += (points[i].y as f64 + points[j].y as f64) * a;
    }
    
    if signed_area.abs() < 1e-6 {
        return None;
    }
    
    signed_area *= 0.5;
    cx /= (6.0 * signed_area);
    cy /= (6.0 * signed_area);
    
    Some([cx, cy])
}

pub fn find_optimal_k(points: &[[f64; 2]], max_k: usize) -> usize {
    if points.len() < 2 {
        return 1;
    }

    let max_k = std::cmp::min(max_k, points.len());
    if max_k < 2 {
        return max_k;
    }

    let mut inertias = Vec::new();
    let k_range: Vec<usize> = (2..=max_k).collect();

    for &k in &k_range {
        let dataset = DatasetBase::new(ndarray::Array2::from(points.to_vec()), ());
        let rng = thread_rng();
        let model = KMeans::params_with_rng(k, rng)
            .max_n_iterations(100)
            .tolerance(1e-5)
            .fit(&dataset);

        match model {
            Ok(model) => {
                inertias.push(model.inertia());
            }
            Err(_) => {
                inertias.push(f64::MAX);
            }
        }
    }

    if inertias.is_empty() {
        return 1;
    }

    // Geometric distance method (Triangle method)
    // Line from first point (p1) to last point (p2)
    let p1_x = k_range[0] as f64;
    let p1_y = inertias[0];
    let p2_x = k_range[k_range.len() - 1] as f64;
    let p2_y = inertias[inertias.len() - 1];

    let mut max_dist = -1.0;
    let mut optimal_k = k_range[0];

    // Vector p2 - p1
    let vec_line_x = p2_x - p1_x;
    let vec_line_y = p2_y - p1_y;
    
    // Normalization factor for the line length
    let line_len = (vec_line_x.powi(2) + vec_line_y.powi(2)).sqrt();

    for (i, &k) in k_range.iter().enumerate() {
        let p3_x = k as f64;
        let p3_y = inertias[i];

        // Vector p1 - p3
        let vec_p1_p3_x = p1_x - p3_x;
        let vec_p1_p3_y = p1_y - p3_y;

        // Cross product magnitude (2D cross product is z-component)
        // (p2-p1) x (p1-p3)
        let cross_prod = vec_line_x * vec_p1_p3_y - vec_line_y * vec_p1_p3_x;
        
        let distance = cross_prod.abs() / line_len;

        if distance > max_dist {
            max_dist = distance;
            optimal_k = k;
        }
    }
    
    optimal_k
}
