CATEGORIES = [
    {
        "id": 0,
        "name": "Depression",
        "min_knot": 0,
        "max_knot": 33,
        "color": "#4CAF50",
        "description": "Disorganized low-level circulation with weak, asymmetric convection.",
    },
    {
        "id": 1,
        "name": "Cyclonic Storm",
        "min_knot": 34,
        "max_knot": 47,
        "color": "#AEEA00",
        "description": "Defined circulation with curved banding and a developing core.",
    },
    {
        "id": 2,
        "name": "Severe Cyclonic Storm",
        "min_knot": 48,
        "max_knot": 63,
        "color": "#FFC107",
        "description": "Tightening core, cold cloud tops, and an emerging central feature.",
    },
    {
        "id": 3,
        "name": "Very Severe Cyclonic Storm",
        "min_knot": 64,
        "max_knot": 89,
        "color": "#FF7043",
        "description": "Symmetric central dense overcast with a visible or warming eye.",
    },
    {
        "id": 4,
        "name": "Extremely Severe Cyclonic Storm",
        "min_knot": 90,
        "max_knot": 119,
        "color": "#E53935",
        "description": "Sharp, well-defined eye surrounded by an intense, symmetric core.",
    },
    {
        "id": 5,
        "name": "Super Cyclonic Storm",
        "min_knot": 120,
        "max_knot": 200,
        "color": "#8E24AA",
        "description": "Pinhole eye with an extremely cold, highly symmetric eyewall.",
    },
]


def knots_to_category(wind_knots: float) -> dict:
    for category in CATEGORIES:
        if category["min_knot"] <= wind_knots <= category["max_knot"]:
            return category
    return CATEGORIES[-1] if wind_knots > CATEGORIES[-1]["max_knot"] else CATEGORIES[0]


def category_probabilities(wind_knots: float, spread: float = 12.0) -> list:
    import math

    midpoints = [(c["min_knot"] + min(c["max_knot"], 160)) / 2 for c in CATEGORIES]
    weights = [math.exp(-((wind_knots - m) ** 2) / (2 * spread ** 2)) for m in midpoints]
    total = sum(weights)
    return [round(w / total, 4) for w in weights]
