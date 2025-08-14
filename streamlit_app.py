# import streamlit as st
# import os, sys, types
# import joblib
# import pandas as pd
# import numpy as np
# from datetime import datetime

# # --- Show runtime versions to debug env mismatches ---
# try:
#     import sklearn, numpy, pandas, joblib as _joblib
#     with st.sidebar:
#         st.caption("Runtime versions")
#         st.code(
#             f"python: 3.11 expected\n"
#             f"sklearn={sklearn.__version__}\n"
#             f"numpy={numpy.__version__}\n"
#             f"pandas={pandas.__version__}\n"
#             f"joblib={_joblib.__version__}",
#             language="text"
#         )
# except Exception:
#     pass

# # ---------------- TimeFeatures transformer ----------------
# RAW_TIME = 'time'
# TIME_FEATURES = ['Hour', 'Day', 'Month']
# BASE_COLS = [
#     'temperature_2m (°C)',
#     'relative_humidity_2m (%)',
#     'dew_point_2m (°C)',
#     'apparent_temperature (°C)',
#     'wind_speed_10m (km/h)',
#     'wind_direction_10m (°)',
#     'cloud_cover (%)'
# ]
# ALL_FEATURES_WITH_TIME = [RAW_TIME] + BASE_COLS + TIME_FEATURES

# class TimeFeatures:
#     """Transformer that extracts Hour/Day/Month from a 'time' column."""
#     def fit(self, X, y=None): return self
#     def transform(self, X):
#         X = X.copy()
#         if RAW_TIME in X.columns:
#             X[RAW_TIME] = pd.to_datetime(X[RAW_TIME])
#             X['Hour']  = X[RAW_TIME].dt.hour
#             X['Day']   = X[RAW_TIME].dt.day
#             X['Month'] = X[RAW_TIME].dt.month
#             X = X.drop(columns=[RAW_TIME])
#         for c in TIME_FEATURES:
#             if c not in X.columns:
#                 X[c] = 0
#         return X
#     def get_params(self, deep=True): return {}
#     def set_params(self, **params): return self

# # Make TimeFeatures available under __main__ for joblib unpickling
# if "__main__" not in sys.modules:
#     sys.modules["__main__"] = types.ModuleType("__main__")
# setattr(sys.modules["__main__"], "TimeFeatures", TimeFeatures)
# # ------------------------------------------------------------

# MODEL_PATH = "model_stacking_pipeline.pkl"

# def _maybe_download_model():
#     """Optional fallback: download model if MODEL_URL env var is set."""
#     url = os.environ.get("MODEL_URL", "").strip()
#     if url and not os.path.exists(MODEL_PATH):
#         import urllib.request
#         st.info("Downloading model…")
#         urllib.request.urlretrieve(url, MODEL_PATH)
#         st.success("Model downloaded.")

# @st.cache_resource
# def load_model():
#     # Try local file first
#     if not os.path.exists(MODEL_PATH):
#         _maybe_download_model()
#     if not os.path.exists(MODEL_PATH):
#         st.error("Model file not found. "
#                  "Ensure `model_stacking_pipeline.pkl` is in the repo (Git LFS), "
#                  "or set an env var MODEL_URL to a direct download link.")
#         st.stop()
#     try:
#         return joblib.load(MODEL_PATH)
#     except Exception as e:
#         st.error(f"Failed to load model: {e}")
#         st.stop()

# model = load_model()

# st.title("☀️ Solar Power Prediction — Batch CSV")

# st.write("Upload a CSV that contains either a `time` column **or** the three columns "
#          "`Hour`, `Day`, `Month`, plus these weather columns:")
# st.code(", ".join(BASE_COLS), language="text")

# file = st.file_uploader("Choose CSV", type=["csv"])
# if file is not None:
#     try:
#         df = pd.read_csv(file)

#         # Ensure time features exist (TimeFeatures in the pipeline will also help)
#         for c in TIME_FEATURES:
#             if c not in df.columns:
#                 df[c] = 0

#         # Keep/align columns; fill any missing weather cols with 0 (StandardScaler can't handle NaN)
#         for col in ALL_FEATURES_WITH_TIME:
#             if col not in df.columns:
#                 df[col] = 0.0

#         X = df[ALL_FEATURES_WITH_TIME]
#         preds = model.predict(X)
#         out = df.copy()
#         out["prediction_W_m2"] = preds

#         st.success(f"Predicted {len(out)} rows.")
#         st.dataframe(out.head(100), use_container_width=True)
#         st.download_button("⬇️ Download predictions.csv",
#                            out.to_csv(index=False).encode("utf-8"),
#                            "predictions.csv",
#                            "text/csv")
#     except Exception as e:
#         st.error(f"Batch prediction error: {e}")


import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dtparse

st.set_page_config(page_title="☀️ Solar Power Prediction — Batch CSV", layout="centered")

# --- 1) Load model/pipeline (trained under scikit-learn==1.2.2) ---
MODEL_PATH = "models/solar_model.pkl"  # <- change to your actual path
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipe = load_model()

# --- 2) Define expected inputs ---
WEATHER_COLS = [
    "temperature_2m",              # °C
    "relative_humidity_2m",        # %
    "dew_point_2m",                # °C
    "apparent_temperature",        # °C
    "wind_speed_10m",              # km/h (will convert to m/s internally)
    "wind_direction_10m",          # °
    "cloud_cover"                  # %
]

TIME_OPTIONS = ["time", ("Hour", "Day", "Month")]  # either a single 'time' or the triple
# Set the exact feature order your pipeline expects (training-time order)
FEATURE_ORDER = [
    # time features first (adjust if your pipeline expects otherwise)
    "Hour", "Day", "Month",
    # then weather features
    *WEATHER_COLS
]

st.title("☀️ Solar Power Prediction — Batch CSV")
st.write("Upload a CSV with either a **time** column or the three columns **Hour, Day, Month**, plus these weather columns:")
st.write(", ".join(WEATHER_COLS))

uploaded = st.file_uploader("Choose CSV", type=["csv"])

def try_parse_time_col(df: pd.DataFrame) -> pd.DataFrame:
    """If a single 'time' column exists, parse it and add Hour/Day/Month."""
    if "time" not in df.columns:
        return df
    # parse robustly
    def _parse(v):
        try:
            return dtparse.parse(str(v))
        except Exception:
            return pd.NaT
    ts = df["time"].map(_parse)
    df["Hour"] = ts.dt.hour
    df["Day"] = ts.dt.day
    df["Month"] = ts.dt.month
    return df

def normalize_units_and_types(df: pd.DataFrame) -> pd.DataFrame:
    # wind speed: if values look like km/h, convert to m/s if your model trained on m/s.
    # If your model was trained on km/h, comment out the conversion below.
    # Here we assume the **incoming CSV is km/h** and the model expects **km/h** (per your UI).
    # -> so no conversion. If needed:
    # df["wind_speed_10m"] = df["wind_speed_10m"] / 3.6
    return df

def validate_and_prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    errors = []

    # If 'time' exists, derive Hour/Day/Month; else expect all three explicitly
    has_time = "time" in df.columns
    has_triple = all(c in df.columns for c in ["Hour", "Day", "Month"])
    if not has_time and not has_triple:
        errors.append("Provide either a 'time' column or the three columns: Hour, Day, Month.")

    if has_time:
        df = try_parse_time_col(df)
        if df[["Hour", "Day", "Month"]].isna().any().any():
            errors.append("Some 'time' values could not be parsed into Hour/Day/Month.")

    # Require all weather columns
    missing_weather = [c for c in WEATHER_COLS if c not in df.columns]
    if missing_weather:
        errors.append(f"Missing required weather columns: {', '.join(missing_weather)}")

    if errors:
        return df, errors

    # Keep only needed columns in the precise order expected by the model
    needed = ["Hour", "Day", "Month"] + WEATHER_COLS
    X = df.copy()

    # Coerce to numeric where appropriate
    for c in needed:
        if c not in X.columns:
            errors.append(f"Column '{c}' missing after parsing.")
        else:
            if c in ["Hour", "Day", "Month"] + WEATHER_COLS:
                X[c] = pd.to_numeric(X[c], errors="coerce")

    # Any NaNs after coercion?
    if X[needed].isna().any().any():
        bad_rows = X[needed].isna().any(axis=1).sum()
        errors.append(f"{bad_rows} row(s) contain non-numeric or missing values in required columns.")

    if errors:
        return X, errors

    X = normalize_units_and_types(X)

    # Final ordering slice; will KeyError if any are missing (already guarded above)
    X = X[FEATURE_ORDER]

    return X, []

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.write("Preview:", raw.head())

    X, errs = validate_and_prepare(raw)
    if errs:
        st.error("Please fix the following problems:")
        for e in errs:
            st.write(f"- {e}")
        st.stop()

    with st.spinner("Predicting…"):
        try:
            y_pred = pipe.predict(X)
        except Exception as e:
            st.error(f"Failed to run the model: {e}")
            st.stop()

    out = raw.copy()
    out["prediction"] = y_pred
    st.success("Done!")
    st.download_button(
        "Download predictions as CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Upload a CSV to start.")

