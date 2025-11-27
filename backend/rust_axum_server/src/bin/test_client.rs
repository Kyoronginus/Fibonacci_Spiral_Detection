use reqwest::multipart;
use serde::Deserialize;
use std::fs::File;
use std::io::{Write, Read};
use base64::{Engine as _, engine::general_purpose};

#[derive(Deserialize)]
struct AnalyzeResponse {
    score: f64,
    b_value: f64,
    image_base64: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = "http://localhost:3000/analyze/";
    let file_path = "../../images/test.jpg";
    let output_path = "../../result.jpg";

    println!("Reading image from {}...", file_path);
    let mut file = File::open(file_path)?;
    let mut file_contents = Vec::new();
    file.read_to_end(&mut file_contents)?;

    let part = multipart::Part::bytes(file_contents)
        .file_name("test.jpg")
        .mime_str("image/jpg")?;

    let form = multipart::Form::new()
        .part("file", part)
        .text("k", "0")
        .text("b_weight", "20000");

    println!("Sending request to {}...", url);
    let client = reqwest::Client::new();
    let response = client.post(url)
        .multipart(form)
        .send()
        .await?;

    if response.status().is_success() {
        let result: AnalyzeResponse = response.json().await?;
        println!("Success!");
        println!("Score: {}", result.score);
        println!("B Value: {}", result.b_value);

        // Remove data:image/png;base64, prefix
        let base64_data = result.image_base64.split(",").nth(1).unwrap_or(&result.image_base64);
        
        let image_data = general_purpose::STANDARD.decode(base64_data)?;
        
        let mut output_file = File::create(output_path)?;
        output_file.write_all(&image_data)?;
        println!("Visualization saved to {}", output_path);
    } else {
        println!("Error: {}", response.status());
        println!("Response: {}", response.text().await?);
    }

    Ok(())
}
