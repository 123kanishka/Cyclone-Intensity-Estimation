const API_BASE = window.location.origin;

const tabButtons = document.querySelectorAll(".tab-button");
const tabContents = document.querySelectorAll(".tab-content");

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    tabButtons.forEach((b) => b.classList.remove("active"));
    tabContents.forEach((c) => c.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`[data-tab-content="${button.dataset.tab}"]`).classList.add("active");
  });
});

const symmetrySlider = document.getElementById("symmetry_score");
const symmetryValue = document.getElementById("symmetry_score_value");
symmetrySlider.addEventListener("input", () => {
  symmetryValue.textContent = symmetrySlider.value;
});

document.getElementById("parameters-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    eye_temperature_c: Number(document.getElementById("eye_temperature_c").value),
    cloud_top_temperature_c: Number(document.getElementById("cloud_top_temperature_c").value),
    eye_diameter_km: Number(document.getElementById("eye_diameter_km").value),
    symmetry_score: Number(document.getElementById("symmetry_score").value),
    latitude: Number(document.getElementById("latitude").value),
    sea_surface_temp_c: Number(document.getElementById("sea_surface_temp_c").value),
  };

  const response = await fetch(`${API_BASE}/api/predict/parameters`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    alert("Could not estimate intensity. Please check the entered values.");
    return;
  }

  renderResult(await response.json());
});

const dropzone = document.getElementById("dropzone");
const imageInput = document.getElementById("image-input");
const dropzoneText = document.getElementById("dropzone-text");
const imagePreview = document.getElementById("image-preview");

dropzone.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    imagePreview.src = reader.result;
    imagePreview.hidden = false;
    dropzoneText.hidden = true;
  };
  reader.readAsDataURL(file);
});

document.getElementById("image-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files[0];
  if (!file) {
    alert("Please select an infrared satellite image first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/api/predict/image`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    alert("Could not analyze the image. Please try a different file.");
    return;
  }

  renderResult(await response.json());
});

function renderResult(result) {
  document.getElementById("empty-state").hidden = true;
  const resultsEl = document.getElementById("results");
  resultsEl.hidden = false;

  document.getElementById("wind-speed-value").textContent = result.wind_speed_knots;
  document.getElementById("wind-speed-kmph").textContent = `${result.wind_speed_kmph} km/h`;
  document.getElementById("central-pressure").textContent = `${result.central_pressure_hpa} hPa`;
  document.getElementById("confidence").textContent = `${Math.round(result.confidence * 100)}%`;

  const badge = document.getElementById("category-badge");
  badge.textContent = "";
  const nameSpan = document.createElement("span");
  nameSpan.id = "category-name";
  nameSpan.textContent = result.category.name;
  badge.appendChild(nameSpan);
  badge.style.background = result.category.color;

  document.getElementById("category-description").textContent = result.category.description;

  drawGauge(result.wind_speed_knots, result.category.color);
  renderProbabilities(result.category_probabilities);
}

function renderProbabilities(probabilities) {
  fetch(`${API_BASE}/api/categories`)
    .then((res) => res.json())
    .then((categories) => {
      const container = document.getElementById("probabilities");
      container.innerHTML = "";
      categories.forEach((category, index) => {
        const probability = probabilities[index] ?? 0;
        const row = document.createElement("div");
        row.className = "probability-row";

        const label = document.createElement("div");
        label.className = "probability-label";
        label.innerHTML = `<span>${category.name}</span><span>${Math.round(probability * 100)}%</span>`;

        const track = document.createElement("div");
        track.className = "probability-bar-track";
        const fill = document.createElement("div");
        fill.className = "probability-bar-fill";
        fill.style.width = `${probability * 100}%`;
        fill.style.background = category.color;
        track.appendChild(fill);

        row.appendChild(label);
        row.appendChild(track);
        container.appendChild(row);
      });
    });
}

function drawGauge(windKnots, color) {
  const canvas = document.getElementById("gauge");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const centerX = width / 2;
  const centerY = height - 10;
  const radius = 110;

  ctx.clearRect(0, 0, width, height);

  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, Math.PI, 2 * Math.PI);
  ctx.lineWidth = 18;
  ctx.strokeStyle = "#1c2942";
  ctx.stroke();

  const maxKnots = 165;
  const fraction = Math.min(windKnots / maxKnots, 1);
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, Math.PI, Math.PI + fraction * Math.PI);
  ctx.lineWidth = 18;
  ctx.strokeStyle = color;
  ctx.lineCap = "round";
  ctx.stroke();
}
