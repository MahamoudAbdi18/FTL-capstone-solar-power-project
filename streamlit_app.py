import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys, types

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
ALL_FEATURES_WITH_TIME = [RAW_TIME] + BASE_COLS + TIME_FEATURES

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
st.caption("Batch prediction of global_tilted_irradiance (W/m²) using a StackingRegressor.")

# ---------------- Batch prediction only ----------------
st.subheader("Upload CSV")
st.write("Your CSV should contain either a `time` column **or** the three columns `Hour`, `Day`, `Month`, plus:")
st.code(", ".join(BASE_COLS), language="text")

file = st.file_uploader("Choose CSV", type=["csv"])
if file is not None:
    try:
        df = pd.read_csv(file)

        for c in TIME_FEATURES:
            if c not in df.columns:
                df[c] = 0
        for col in ALL_FEATURES_WITH_TIME:
            if col not in df.columns:
                df[col] = np.nan
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
