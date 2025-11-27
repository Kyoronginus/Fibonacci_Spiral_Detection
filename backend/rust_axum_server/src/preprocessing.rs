use image::{DynamicImage, imageops::FilterType};

pub fn smart_resize(image: &DynamicImage, max_dim: u32) -> DynamicImage {
    let (w, h) = (image.width(), image.height());

    if w <= max_dim && h <= max_dim {
        return image.clone();
    }

    let (new_w, new_h) = if w > h {
        let new_w = max_dim;
        let new_h = (h as f64 * (max_dim as f64 / w as f64)) as u32;
        (new_w, new_h)
    } else {
        let new_h = max_dim;
        let new_w = (w as f64 * (max_dim as f64 / h as f64)) as u32;
        (new_w, new_h)
    };

    image.resize(new_w, new_h, FilterType::Lanczos3)
}
