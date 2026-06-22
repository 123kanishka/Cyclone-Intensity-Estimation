import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.model.labels import CATEGORIES  # noqa: E402


def wind_speed_to_category_id(wind_knots: float) -> int:
    for category in CATEGORIES:
        if category["min_knot"] <= wind_knots <= category["max_knot"]:
            return category["id"]
    return CATEGORIES[-1]["id"] if wind_knots > CATEGORIES[-1]["max_knot"] else CATEGORIES[0]["id"]
