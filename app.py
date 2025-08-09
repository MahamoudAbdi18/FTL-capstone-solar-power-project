
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("model_stacking_pipeline.pkl")

app = FastAPI(title="Solar Stacking Regressor API", version="1.0.0")

class SolarFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    time: Optional[str] = Field(default=None, description="ISO datetime, e.g., 2025-08-09T12:00:00")

    temperature_2m_c: float = Field(alias="temperature_2m (°C)")
    relative_humidity_2m_pct: float = Field(alias="relative_humidity_2m (%)")
    dew_point_2m_c: float = Field(alias="dew_point_2m (°C)")
    apparent_temperature_c: float = Field(alias="apparent_temperature (°C)")
    wind_speed_10m_kmh: float = Field(alias="wind_speed_10m (km/h)")
    wind_direction_10m_deg: float = Field(alias="wind_direction_10m (°)")
    cloud_cover_pct: float = Field(alias="cloud_cover (%)")

    Hour: Optional[int] = None
    Day: Optional[int] = None
    Month: Optional[int] = None

def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("Model file not found. Train first with train_pipeline.py")
    return joblib.load(MODEL_PATH)

model = load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: SolarFeatures):
    data = payload.dict(by_alias=True)

    row = {
        'time': data.get('time'),
        'temperature_2m (°C)': data.get('temperature_2m (°C)', data.get('temperature_2m_c')),
        'relative_humidity_2m (%)': data.get('relative_humidity_2m (%)', data.get('relative_humidity_2m_pct')),
        'dew_point_2m (°C)': data.get('dew_point_2m (°C)', data.get('dew_point_2m_c')),
        'apparent_temperature (°C)': data.get('apparent_temperature (°C)', data.get('apparent_temperature_c')),
        'wind_speed_10m (km/h)': data.get('wind_speed_10m (km/h)', data.get('wind_speed_10m_kmh')),
        'wind_direction_10m (°)': data.get('wind_direction_10m (°)', data.get('wind_direction_10m_deg')),
        'cloud_cover (%)': data.get('cloud_cover (%)', data.get('cloud_cover_pct')),
        'Hour': data.get('Hour'),
        'Day': data.get('Day'),
        'Month': data.get('Month'),
    }
    X = pd.DataFrame([row])
    try:
        y_pred = model.predict(X)[0]
        return {"prediction_W_m2": float(y_pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
