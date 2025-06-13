# Fibonacci Spiral Detection

A web application that analyzes images to detect and visualize Fibonacci spirals in compositions. The app identifies key points in an image, clusters them, and fits a logarithmic spiral that approximates the golden ratio.

![Fibonacci Spiral Detection App](https://via.placeholder.com/800x400?text=Fibonacci+Spiral+Detection+Screenshot)

## Architecture

The application uses a three-tier architecture:

1. **Frontend**: Static HTML/CSS/JS hosted on Firebase
2. **Rust Server**: Acts as an entrypoint/proxy, handling initial requests and file uploads
3. **Python Service**: Performs the image analysis, clustering, and spiral fitting

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend  │────▶│   Rust Server   │────▶│  Python Service │
│  (Firebase) │     │  (Cloud Run)    │     │   (Cloud Run)   │
└─────────────┘     └─────────────────┘     └─────────────────┘
```

## Key Features

- Upload and analyze images to detect Fibonacci spiral patterns
- Adjustable clustering parameter (k) to control point grouping
- Customizable spiral fitting with b-weight parameter
- Real-time preview of detected clusters
- Scoring system that evaluates how closely the composition follows the golden ratio
- Visualization of the detected spiral overlaid on the original image

## Technologies Used

### Frontend
- HTML5, CSS3, JavaScript
- Firebase Hosting

### Backend
- **Rust Server**:
  - Axum web framework
  - Tokio async runtime
  - Reqwest for HTTP client
  - Tower-HTTP for middleware

- **Python Service**:
  - FastAPI
  - OpenCV for image processing
  - scikit-learn for clustering
  - NumPy for numerical operations
  - Matplotlib for visualization

### Deployment
- Google Cloud Run for containerized services
- Docker for containerization
- GitHub Actions for CI/CD

## Usage

1. Open the application in your browser
2. Upload an image containing a composition you want to analyze
3. Adjust the k-value slider if you want to control the number of clusters
4. Adjust the b-weight slider to control how strictly the spiral follows the golden ratio
5. Click "Analyze" to process the image
6. View the results showing:
   - The original image with the detected spiral overlay
   - A score indicating how closely the composition follows the golden ratio
   - The detected b-value and how it compares to the golden ratio

## Troubleshooting

### Common Issues

1. **502 Bad Gateway Error**:
   - Check if both backend services are running
   - Verify the Python service URL in the Rust server configuration
   - Ensure the Python service allows unauthenticated access

2. **Image Analysis Fails**:
   - Ensure the image has sufficient contrast for object detection
   - Try adjusting the k-value to control clustering
   - Images should be under 10MB for optimal performance

## Author

Kyoronginus - [kyoronginus@gmail.com](mailto:kyoronginus@gmail.com)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The golden ratio (φ ≈ 1.618) and Fibonacci sequence for their mathematical beauty
- OpenCV community for the excellent computer vision tools
- Rust and Python communities for their amazing frameworks