"""
Inference entry points for both supported input modes.

If a trained checkpoint is present at MODEL_WEIGHTS_PATH, predictions route
through the dual-input CNN in `architecture.py`. Otherwise both endpoints fall
back to the deterministic estimator in `heuristic.py`, keeping the API fully
functional without a GPU or a trained checkpoint.
"""

import os
from io import BytesIO

import numpy as np
from PIL import Image

from . import heuristic

MODEL_WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS_PATH", "")
IMAGE_SIZE = (256, 256)

_trained_model = None


def _try_load_trained_model():
    global _trained_model
    if _trained_model is not None or not MODEL_WEIGHTS_PATH:
        return _trained_model
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        return None
    try:
        from .architecture import build_intensity_model

        model = build_intensity_model()
        model.load_weights(MODEL_WEIGHTS_PATH)
        _trained_model = model
    except Exception:
        _trained_model = None
    return _trained_model


def predict_from_parameters(params: dict) -> dict:
    result = heuristic.estimate_from_parameters(**params)
    result["source"] = "parameter_heuristic"
    return result


def predict_from_image(image_bytes: bytes) -> dict:
    model = _try_load_trained_model()
    image_array = _load_grayscale(image_bytes)

    if model is not None:
        return _predict_with_model(model, image_array)

    coldest, warmest, symmetry, texture_std = _image_statistics(image_array)
    result = heuristic.estimate_from_image_stats(
        coldest_region_intensity=coldest,
        warm_core_intensity=warmest,
        symmetry_score=symmetry,
        texture_std=texture_std,
    )
    result["source"] = "image_heuristic"
    return result


def _load_grayscale(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("L").resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32)


def _image_statistics(image_array: np.ndarray):
    coldest_region_intensity = float(np.percentile(image_array, 2))
    warm_core_intensity = float(np.percentile(image_array, 98))

    left, right = np.array_split(image_array, 2, axis=1)
    right_flipped = np.flip(right, axis=1)
    min_width = min(left.shape[1], right_flipped.shape[1])
    correlation = np.corrcoef(left[:, :min_width].flatten(), right_flipped[:, :min_width].flatten())[0, 1]
    symmetry_score = float(np.clip((correlation + 1) * 50, 0, 100))

    texture_std = float(np.std(image_array))

    return coldest_region_intensity, warm_core_intensity, symmetry_score, texture_std


def _predict_with_model(model, image_array: np.ndarray) -> dict:
    normalized = (image_array / 255.0)[np.newaxis, ..., np.newaxis]
    default_metadata = np.zeros((1, 6), dtype=np.float32)

    category_probs, wind_speed = model.predict([normalized, default_metadata], verbose=0)
    wind_knots = float(wind_speed[0][0])

    from .labels import knots_to_category

    return {
        "wind_speed_knots": round(wind_knots, 1),
        "wind_speed_kmph": round(wind_knots * 1.852, 1),
        "central_pressure_hpa": round(1010 - 0.112 * (wind_knots ** 1.35), 1),
        "category": knots_to_category(wind_knots),
        "category_probabilities": [round(float(p), 4) for p in category_probs[0]],
        "confidence": float(np.max(category_probs[0])),
        "source": "trained_model",
    }
