use image::{DynamicImage, Rgba};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut};
use crate::spiral_fit::SpiralParams;
use anyhow::Result;

pub fn draw_result(
    image: &DynamicImage,
    initial_centroids: &[[f64; 2]],
    clustered_centroids: &[[f64; 2]],
    spiral_params: Option<&SpiralParams>
) -> Result<DynamicImage> {
    let mut vis_image = image.clone();

    // Draw initial centroids (Red)
    for point in initial_centroids {
        draw_filled_circle_mut(
            &mut vis_image,
            (point[0] as i32, point[1] as i32),
            1,
            Rgba([150, 150, 150, 255]),
        );
    }

    // Draw clustered centroids (Green)
    for point in clustered_centroids {
        draw_filled_circle_mut(
            &mut vis_image,
            (point[0] as i32, point[1] as i32),
            5,
            Rgba([255, 0, 0, 255]),
        );
    }

    // Draw Spiral (Blue)
    if let Some(params) = spiral_params {
        let theta_max = 12.0 * std::f64::consts::PI;
        let num_points = 10000;
        let mut prev_point: Option<(f32, f32)> = None;

        for i in 0..num_points {
            let t = (i as f64 / (num_points - 1) as f64); // 0.0 to 1.0
            let theta = -theta_max + (2.0 * theta_max) * t; // -theta_max to +theta_max
            let r = params.a * (params.b * theta).exp();
            let x = params.cx + r * theta.cos();
            let y = params.cy + r * theta.sin();

            let current_point = (x as f32, y as f32);

            if let Some(prev) = prev_point {
                let color = Rgba([0, 0, 255, 255]);
                // Draw the line segment multiple times with small offsets to simulate thickness
                draw_line_segment_mut(&mut vis_image, prev, current_point, color);
                draw_line_segment_mut(
                    &mut vis_image,
                    (prev.0 + 1.0, prev.1),
                    (current_point.0 + 1.0, current_point.1),
                    color,
                );
                draw_line_segment_mut(
                    &mut vis_image,
                    (prev.0, prev.1 + 1.0),
                    (current_point.0, current_point.1 + 1.0),
                    color,
                );
            }
            prev_point = Some(current_point);
        }
    }

    Ok(vis_image)
}
