// Fibonacci Spiral Detection - Main Script

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const resultBox = document.getElementById("image-result-box");
const spinner = document.getElementById("spinner");
const kSlider = document.getElementById("k-slider");
const kValueLabel = document.getElementById("k-value-label");
const bWeightSlider = document.getElementById("b-weight-slider");
const bWeightLabel = document.getElementById("b-weight-label");

// Debounce function to wait for user to finish moving slider
function debounce(func, delay) {
  let timeout;
  return function (...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), delay);
  };
}

// Fetch preview image asynchronously
const fetchPreview = async () => {
  const file = fileInput.files[0];
  const kValue = kSlider.value;

  // Don't fetch preview if k=0 (Auto) or no file selected
  if (kValue === "0" || !file) {
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("k", kValue);

  try {
    const response = await fetch(`${import.meta.env.VITE_API_URL}/preview_clusters/`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) return; // Silently fail on error

    const imageBlob = await response.blob();
    const imageUrl = URL.createObjectURL(imageBlob);

    // Update existing preview image or create new one
    let imgElement = resultBox.querySelector("img");
    if (!imgElement) {
      imgElement = document.createElement("img");
      resultBox.innerHTML = "";
      resultBox.appendChild(imgElement);
    }
    imgElement.src = imageUrl;
  } catch (error) {
    console.error("Preview error:", error);
  }
};

// Debounced preview function with 300ms delay
const debouncedFetchPreview = debounce(fetchPreview, 300);

// Update k-value label and fetch preview when slider moves
kSlider.addEventListener("input", (event) => {
  const value = event.target.value;
  kValueLabel.textContent = value === "0" ? "Auto" : value;
  debouncedFetchPreview();
});

// Update preview when file changes
fileInput.addEventListener("change", () => {
  resultBox.innerHTML = "Select K value for preview.";
  debouncedFetchPreview();
});

// Update b-weight label when slider moves
bWeightSlider.addEventListener("input", (event) => {
  const value = parseInt(event.target.value, 10);
  let label = "Medium";
  if (value < 500) label = "None";
  else if (value < 5000) label = "Low";
  else if (value > 15000) label = "Very High";
  bWeightLabel.textContent = label;
});

// Main analysis form submission
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    return;
  }

  const kValue = kSlider.value;
  const bWeightValue = bWeightSlider.value;

  const formData = new FormData();
  formData.append("file", file);
  formData.append("k", kValue);
  formData.append("b_weight", bWeightValue);

  spinner.style.display = "block";
  resultBox.innerHTML = "";

  try {
    const response = await fetch(`${import.meta.env.VITE_API_URL}/analyze/`, {
      method: "POST",
      body: formData,
    });
    spinner.style.display = "none";

    if (!response.ok) {
      const errorText = await response.text();
      resultBox.innerHTML = "Error: " + errorText;
      return;
    }
    const result = await response.json();
    // Display result image and score
    resultBox.innerHTML = `
            <div>
                <img src="${result.image_base64}" alt="Result Image" />
            </div>
        `;
  } catch (error) {
    spinner.style.display = "none";
    resultBox.innerHTML = "Error: " + error;
  }
});
