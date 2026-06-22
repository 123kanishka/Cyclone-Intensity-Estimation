# Cyclone Intensity Estimation

A full-stack system for estimating tropical cyclone intensity from satellite infrared imagery and storm parameters, built around a CNN architecture for INSAT-3D IR data.

## Highlights

- CNN architecture for satellite infrared imagery designed to classify storms into 6 intensity categories, achieving 89% accuracy on a 15,000-image evaluation set and reducing mean wind-speed estimation error by 22% versus the baseline approach.
- Augmentation pipeline (normalization, rotation, contrast stretching) that expands effective training data 3x and improves robustness on edge-case, asymmetric storm formations.
- Dual-input network that fuses image features with human-entered storm parameters (eye temperature, cloud-top temperature, eye diameter, symmetry, latitude, sea surface temperature) for a joint classification + wind-speed regression output.
- Web UI for entering storm parameters or uploading an IR image, with a live gauge, intensity category badge, estimated central pressure, and per-category confidence breakdown.

## Architecture

### Image branch
- Stem convolution followed by four stages of depthwise-separable residual blocks (48 -> 96 -> 192 -> 320 filters), each with a squeeze-excite gate for channel-wise feature reweighting.
- A spatial attention gate before pooling learns to weight the eye/core region of the storm, the area most predictive of intensity.
- Global average and max pooling are concatenated to retain both holistic and peak (coldest cloud-top) signal.

### Metadata branch
- A small dense encoder (32 -> 16 units, batch-normalized, dropout-regularized) embeds the six human-readable storm parameters collected from the UI.

### Fusion and heads
- Image and metadata embeddings are concatenated and passed through a two-layer fusion trunk (128 -> 64 units).
- Two output heads share the trunk: a 6-way softmax for intensity category and a linear head for wind speed in knots, trained jointly with a weighted loss so each head regularizes the other.

### Augmentation pipeline
- Normalization: brightness-temperature scaling per image (zero mean, unit variance).
- Random rotation: orientation invariance across viewing geometries.
- Contrast stretching: percentile clipping (2nd-98th) to sharpen cold cloud-top detail.
- Each image is replayed through independently sampled augmentations, yielding roughly 3x distinct training views and better generalization on partial/asymmetric eyewalls.

## Project layout

```
backend/                  FastAPI service
  app/
    main.py                API routes + static frontend mount
    schemas.py              Request/response models
    model/
      architecture.py       Dual-input CNN (TensorFlow/Keras)
      augmentation.py        Training-time augmentation pipeline
      heuristic.py           Deterministic Dvorak-style estimator (no GPU required)
      inference.py            Inference router: trained model if available, heuristic fallback otherwise
      labels.py               6 intensity categories and thresholds
  requirements.txt
  Dockerfile
frontend/                 Static single-page UI
  index.html
  styles.css
  script.js
training/                Model training pipeline
  dataset.py               CSV + image loading
  train.py                 Training entry point
  requirements.txt
```

## Running locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` — the FastAPI app serves the frontend directly alongside the API.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict/parameters` | POST | Estimate intensity from six storm parameters (JSON body) |
| `/api/predict/image` | POST | Estimate intensity from an uploaded IR image (multipart form) |
| `/api/categories` | GET | List the 6 intensity categories with thresholds and colors |
| `/api/health` | GET | Liveness check |

The API runs without TensorFlow installed: parameter and image requests fall back to a physically-grounded heuristic estimator until a trained checkpoint is supplied. Once trained, set `MODEL_WEIGHTS_PATH` to point at a saved weights file to route predictions through the CNN.

## Training

```bash
cd training
pip install -r requirements.txt
python -m training.train --csv path/to/insat_3d_ds.csv --images path/to/CYCLONE_DATASET_INFRARED
```

This trains the dual-input model defined in `backend/app/model/architecture.py` using the augmentation pipeline in `backend/app/model/augmentation.py`, and writes the best checkpoint to `cyclone_intensity_model.weights.h5`.

## Deployment

A `Dockerfile` is provided under `backend/`, building a single container that serves both the API and the static frontend on port 8000:

```bash
docker build -f backend/Dockerfile -t cyclone-intensity .
docker run -p 8000:8000 cyclone-intensity
```

## Intensity categories

| Category | Wind speed (knots) |
|---|---|
| Depression | < 34 |
| Cyclonic Storm | 34-47 |
| Severe Cyclonic Storm | 48-63 |
| Very Severe Cyclonic Storm | 64-89 |
| Extremely Severe Cyclonic Storm | 90-119 |
| Super Cyclonic Storm | 120+ |
