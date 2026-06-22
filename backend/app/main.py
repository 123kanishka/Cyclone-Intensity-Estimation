from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .model import inference
from .model.labels import CATEGORIES
from .schemas import ParameterInput, PredictionResponse

app = FastAPI(title="Cyclone Intensity Estimation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/categories")
def get_categories():
    return CATEGORIES


@app.post("/api/predict/parameters", response_model=PredictionResponse)
def predict_parameters(payload: ParameterInput):
    return inference.predict_from_parameters(payload.model_dump())


@app.post("/api/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    return inference.predict_from_image(image_bytes)


frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
