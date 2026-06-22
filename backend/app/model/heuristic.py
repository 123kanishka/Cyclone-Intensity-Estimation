"""
Dvorak-inspired estimator used when no trained model checkpoint is present.

The CNN defined in `model/architecture.py` is the production path once trained
on labelled INSAT-3D imagery (see `training/train.py`). This module provides a
physically-grounded fallback so the API and UI are fully functional out of the
box, using the same meteorological cues (eye-to-cloud thermal gradient, core
symmetry, eye size, latitude) the CNN ultimately learns to extract from pixels.
"""

from .labels import knots_to_category, category_probabilities


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def estimate_from_parameters(
    eye_temperature_c: float,
    cloud_top_temperature_c: float,
    eye_diameter_km: float,
    symmetry_score: float,
    latitude: float,
    sea_surface_temp_c: float = 28.0,
) -> dict:
    thermal_gradient = eye_temperature_c - cloud_top_temperature_c

    gradient_term = _clip(thermal_gradient, -10, 60) * 1.55
    symmetry_term = _clip(symmetry_score, 0, 100) * 0.42
    eye_term = max(0.0, 45 - _clip(eye_diameter_km, 5, 80)) * 0.55
    sst_term = max(0.0, sea_surface_temp_c - 26.0) * 1.8
    latitude_penalty = max(0.0, abs(latitude) - 25) * 1.1

    wind_knots = 18 + gradient_term + symmetry_term + eye_term + sst_term - latitude_penalty
    wind_knots = round(_clip(wind_knots, 10, 165), 1)

    return _build_response(wind_knots)


def estimate_from_image_stats(
    coldest_region_intensity: float,
    warm_core_intensity: float,
    symmetry_score: float,
    texture_std: float,
) -> dict:
    pixel_gradient = warm_core_intensity - coldest_region_intensity

    gradient_term = _clip(pixel_gradient, 0, 220) * 0.42
    symmetry_term = _clip(symmetry_score, 0, 100) * 0.48
    texture_term = _clip(texture_std, 0, 90) * 0.25

    wind_knots = 20 + gradient_term + symmetry_term + texture_term
    wind_knots = round(_clip(wind_knots, 10, 165), 1)

    return _build_response(wind_knots)


def _build_response(wind_knots: float) -> dict:
    category = knots_to_category(wind_knots)
    central_pressure_hpa = round(1010 - 0.112 * (wind_knots ** 1.35), 1)
    probabilities = category_probabilities(wind_knots)

    return {
        "wind_speed_knots": wind_knots,
        "wind_speed_kmph": round(wind_knots * 1.852, 1),
        "central_pressure_hpa": central_pressure_hpa,
        "category": category,
        "category_probabilities": probabilities,
        "confidence": max(probabilities),
    }
