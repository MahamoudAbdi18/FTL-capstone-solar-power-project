import streamlit as st
import os, sys, types
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# --- Show runtime versions to debug env mismatches ---
try:
    import sklearn, numpy, pandas, joblib as _joblib
    with st.sidebar:
        st.caption("Runtime versions")
        st.code(
            f"python: 3.11 expected\n"
            f"sklearn={sklearn.__version__}\n"
            f"numpy={numpy.__version__}\n"
            f"pandas={pandas.__version__}\n"
            f"joblib={_joblib.__version__}",
            language="text"
        )
except Exception:
    pass

# ---------------- TimeFeatures transformer ----------------
RAW_TIME = 'time'
TIME_FEATURES = ['Hour', 'Day', 'Month']
BASE_COLS = [
    'temperature_2m (°C)',
    'relative_humidity_2m (%)',
    'dew_point_2m (°C)',
    'wind_speed_10m (km/h)',
    'wind_direction_10m (°)',
    'cloud_cover (%)'
]
ALL_FEATURES_WITH_TIME = [RAW_TIME] + BASE_COLS + TIME_FEATURES

class TimeFeatures:
    """Transformer that extracts Hour/Day/Month from a 'time' column."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if RAW_TIME in X.columns:
            X[RAW_TIME] = pd.to_datetime(X[RAW_TIME])
            X['Hour']  = X[RAW_TIME].dt.hour
            X['Day']   = X[RAW_TIME].dt.day
            X['Month'] = X[RAW_TIME].dt.month
            X = X.drop(columns=[RAW_TIME])
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

MODEL_PATH = "model_stacking_pipeline.pkl"

def _maybe_download_model():
    """Optional fallback: download model if MODEL_URL env var is set."""
    url = os.environ.get("MODEL_URL", "").strip()
    if url and not os.path.exists(MODEL_PATH):
        import urllib.request
        st.info("Downloading model…")
        urllib.request.urlretrieve(url, MODEL_PATH)
        st.success("Model downloaded.")

@st.cache_resource
def load_model():
    # Try local file first
    if not os.path.exists(MODEL_PATH):
        _maybe_download_model()
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. "
                 "Ensure `model_stacking_pipeline.pkl` is in the repo (Git LFS), "
                 "or set an env var MODEL_URL to a direct download link.")
        st.stop()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_model()




st.title("☀️ Solar Power Prediction — Saisie manuelle")

st.write("Renseignez soit un **horaire** (champ unique), soit **Heure/Jour/Mois**, "
         "puis les variables météo suivantes :")
st.code(", ".join(BASE_COLS), language="text")

# --- Choix de la méthode de temps ---
time_mode = st.radio(
    "Comment fournir le temps ?",
    options=["Colonne 'time' (unique)", "Heure / Jour / Mois"],
    horizontal=True,
)

with st.form("manual_input_form"):
    st.subheader("Temps")
    if time_mode == "Colonne 'time' (unique)":
        date_val = st.date_input("Date")
        time_val = st.time_input("Heure")
        dt = pd.to_datetime(f"{date_val} {time_val}")
        hour = day = month = None
    else:
        col_h, col_d, col_m = st.columns(3)
        with col_h:
            hour = st.number_input("Hour", min_value=0, max_value=23, value=12, step=1)
        with col_d:
            day = st.number_input("Day", min_value=1, max_value=31, value=15, step=1)
        with col_m:
            month = st.number_input("Month", min_value=1, max_value=12, value=6, step=1)
        dt = None

    st.subheader("Variables météo")
    c1, c2 = st.columns(2)
    with c1:
        temperature_2m = st.number_input("temperature_2m (°C)", value=25.0)
        relative_humidity_2m = st.number_input("relative_humidity_2m (%)", min_value=0.0, max_value=100.0, value=50.0)
        dew_point_2m = st.number_input("dew_point_2m (°C)", value=15.0)
    with c2:
        wind_speed_10m = st.number_input("wind_speed_10m (km/h)", min_value=0.0, value=10.0)
        wind_direction_10m = st.number_input("wind_direction_10m (°)", min_value=0.0, max_value=360.0, value=180.0)
        cloud_cover = st.number_input("cloud_cover (%)", min_value=0.0, max_value=100.0, value=20.0)

    submitted = st.form_submit_button("Prédire")

if submitted:
    row = {}
    if time_mode == "Colonne 'time' (unique)":
        row[RAW_TIME] = dt
        row["Hour"] = 0
        row["Day"] = 0
        row["Month"] = 0
    else:
        row["Hour"] = int(hour)
        row["Day"] = int(day)
        row["Month"] = int(month)
        row[RAW_TIME] = pd.NaT

    row['temperature_2m (°C)'] = float(temperature_2m)
    row['relative_humidity_2m (%)'] = float(relative_humidity_2m)
    row['dew_point_2m (°C)'] = float(dew_point_2m)
    row['wind_speed_10m (km/h)'] = float(wind_speed_10m)
    row['wind_direction_10m (°)'] = float(wind_direction_10m)
    row['cloud_cover (%)'] = float(cloud_cover)

    for col in ALL_FEATURES_WITH_TIME:
        row.setdefault(col, 0 if col in TIME_FEATURES else 0.0)

    X = pd.DataFrame([row], columns=ALL_FEATURES_WITH_TIME)

    try:
        y = model.predict(X)
        pred = float(y[0])
        st.success("✅ Prédiction effectuée")
        st.metric("Prediction (W/m²)", f"{pred:,.2f}")
        out = X.copy()
        out["prediction_W_m2"] = pred
        st.download_button(
            "⬇️ Télécharger la prédiction (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="prediction_unique.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Erreur de prédiction : {e}")

st.title("☀️ Solar Power Prediction ")

st.write("Upload a CSV that contains either a `time` column **or** the three columns "
         "`Hour`, `Day`, `Month`, plus these weather columns:")
st.code(", ".join(BASE_COLS), language="text")

file = st.file_uploader("Choose CSV", type=["csv"])
if file is not None:
    try:
        df = pd.read_csv(file)

        # Ensure time features exist (TimeFeatures in the pipeline will also help)
        for c in TIME_FEATURES:
            if c not in df.columns:
                df[c] = 0

        # Keep/align columns; fill any missing weather cols with 0 (StandardScaler can't handle NaN)
        for col in ALL_FEATURES_WITH_TIME:
            if col not in df.columns:
                df[col] = 0.0

        X = df[ALL_FEATURES_WITH_TIME]
        preds = model.predict(X)
        out = df.copy()
        out["prediction_W_m2"] = preds

        st.success(f"Predicted {len(out)} rows.")
        st.dataframe(out.head(100), use_container_width=True)
        st.download_button("⬇️ Download predictions.csv",
                           out.to_csv(index=False).encode("utf-8"),
                          "predictions.csv",
                          "text/csv")
    except Exception as e:
        st.error(f"Batch prediction error: {e}")



