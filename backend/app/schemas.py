from pydantic import BaseModel, Field


class ParameterInput(BaseModel):
    eye_temperature_c: float = Field(..., ge=-90, le=30, description="Estimated brightness temperature at the storm center, in Celsius")
    cloud_top_temperature_c: float = Field(..., ge=-90, le=30, description="Coldest surrounding cloud-top brightness temperature, in Celsius")
    eye_diameter_km: float = Field(..., ge=0, le=120, description="Approximate eye or central feature diameter, in kilometers")
    symmetry_score: float = Field(..., ge=0, le=100, description="Visual symmetry of the cloud pattern around the center, 0-100")
    latitude: float = Field(..., ge=-60, le=60, description="Storm center latitude in degrees, negative for southern hemisphere")
    sea_surface_temp_c: float = Field(28.0, ge=15, le=35, description="Sea surface temperature beneath the storm, in Celsius")


class CategoryInfo(BaseModel):
    id: int
    name: str
    min_knot: int
    max_knot: int
    color: str
    description: str


class PredictionResponse(BaseModel):
    wind_speed_knots: float
    wind_speed_kmph: float
    central_pressure_hpa: float
    confidence: float
    category: CategoryInfo
    category_probabilities: list[float]
    source: str
