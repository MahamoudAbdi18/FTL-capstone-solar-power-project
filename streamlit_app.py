import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys, types
from datetime import datetime

# ---------------- TimeFeatures transformer ----------------
RAW_TIME = 'time'
TIME_FEATURES = ['Hour', 'Day', 'Month']
BASE_COLS = [
    'temperature_2m (°C)',
    'relative_humidity_2m (%)',
    'dew_point_2m (°C)',
    'apparent_temperature (°C)',
    'wind_speed_10m (km/h)',
    'wind_direction_10m (°)',
    'cloud_cover (%)'
]
ALL_FEATURES_WITH_TIME = [RAW_TIME] + BASE_COLS + TIME_FEATURES   # matches training pipeline

class TimeFeatures:
    """Transformer that extracts Hour/Day/Month from a 'time' column."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if RAW_TIME in X.columns:
            X[RAW_TIME] = pd.to_datetime(X[RAW_TIME])
            X['Hour']  = X[RAW_TIME].dt.hour
            X['Day']   = X[RAW_TIME].dt.day
            X['Month'] = X[RAW_TIME].dt.month
            X = X.drop(columns=[RAW_TIME])
        # ensure expected columns
        for c in TIME_FEATURES:
            if c not in X.columns:
                X[c] = 0
        return X

    def get_params(self, deep=True): return {}
    def set_params(self, **params): return self

# Make TimeFeatures available under __main__ for joblib unpickling
if "__main__" not in sys.modules:
    sys.modules["__main__"] = types.ModuleType("__main__")
setattr(sys.modules["__main__"], "TimeFeatures", TimeFeatures)
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("model_stacking_pipeline.pkl")

model = load_model()

st.title("☀️ Solar Power Prediction Dashboard")
st.caption("Predicts **global_tilted_irradiance (W/m²)** using a StackingRegressor.")

tab1, tab2 = st.tabs(["Single prediction", "Batch prediction (CSV)"])

# ---------------- Single prediction ----------------
with tab1:
    st.subheader("Inputs")

    use_time = st.toggle("Provide exact datetime (ISO 8601)", value=True,
                         help="If off, set Hour/Day/Month manually.")
    time_str = None
    hour = day = month = None

    if use_time:
        time_str = st.text_input(
            "Datetime (ISO, e.g. 2025-08-09T12:00:00)",
            value=datetime.utcnow().strftime("%Y-%m-%dT%H:00:00")
        )
    else:
        c1, c2, c3 = st.columns(3)
        with c1: hour  = st.number_input("Hour", 0, 23, 12)
        with c2: day   = st.number_input("Day", 1, 31, 15)
        with c3: month = st.number_input("Month", 1, 12, 6)

    c1, c2, c3 = st.columns(3)
    with c1:
        temperature   = st.number_input("temperature_2m (°C)", value=25.0)
        wind_speed    = st.number_input("wind_speed_10m (km/h)", value=5.0)
        cloud_cover   = st.number_input("cloud_cover (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    with c2:
        humidity      = st.number_input("relative_humidity_2m (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        wind_dir      = st.number_input("wind_direction_10m (°)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)
    with c3:
        dew_point     = st.number_input("dew_point_2m (°C)", value=10.0)
        apparent_temp = st.number_input("apparent_temperature (°C)", value=25.0)

    if st.button("🔮 Predict"):
        # Build a single-row DataFrame with exactly the columns your pipeline expects
        row = {
            'time': time_str if use_time else None,
            'temperature_2m (°C)': temperature,
            'relative_humidity_2m (%)': humidity,
            'dew_point_2m (°C)': dew_point,
            'apparent_temperature (°C)': apparent_temp,
            'wind_speed_10m (km/h)': wind_speed,
            'wind_direction_10m (°)': wind_dir,
            'cloud_cover (%)': cloud_cover,
            'Hour': hour, 'Day': day, 'Month': month
        }
        X = pd.DataFrame([row], columns=ALL_FEATURES_WITH_TIME)  # keep column order consistent

        try:
            y = model.predict(X)[0]
            st.success(f"Predicted global_tilted_irradiance: **{y:.2f} W/m²**")
            st.dataframe(X, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------- Batch prediction ----------------
with tab2:
    st.subheader("Upload CSV")
    st.write("CSV should contain either a `time` column **or** the three columns `Hour`, `Day`, `Month`, plus:")
    st.code(", ".join(BASE_COLS), language="text")

    file = st.file_uploader("Choose CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)

            # Ensure necessary columns exist; create missing time features as 0 (the transformer will also handle 'time')
            for c in TIME_FEATURES:
                if c not in df.columns:
                    df[c] = 0
            # Keep only expected columns (extras are ignored)
            for col in ALL_FEATURES_WITH_TIME:
                if col not in df.columns:
                    df[col] = np.nan  # fill missing base cols as NaN; scaler/model can handle if trained accordingly
            X = df[ALL_FEATURES_WITH_TIME]

            preds = model.predict(X)
            out = df.copy()
            out['prediction_W_m2'] = preds
            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out.head(100), use_container_width=True)
            st.download_button("⬇️ Download predictions.csv",
                               out.to_csv(index=False).encode("utf-8"),
                               "predictions.csv",
                               "text/csv")
        except Exception as e:
            st.error(f"Batch prediction error: {e}")
