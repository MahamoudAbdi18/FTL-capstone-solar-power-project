import streamlit as st
import joblib
import pandas as pd
import numpy as np

# time_features.py
import pandas as pd

RAW_TIME = 'time'
TIME_FEATURES = ['Hour', 'Day', 'Month']

class TimeFeatures:
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if RAW_TIME in X.columns:
            X[RAW_TIME] = pd.to_datetime(X[RAW_TIME])
            X['Hour'] = X[RAW_TIME].dt.hour
            X['Day']  = X[RAW_TIME].dt.day
            X['Month']= X[RAW_TIME].dt.month
            X = X.drop(columns=[RAW_TIME])
        for c in TIME_FEATURES:
            if c not in X.columns:
                X[c] = 0
        return X
    def get_params(self, deep=True): return {}
    def set_params(self, **params): return self


# Create a fake __main__ module and attach TimeFeatures to it,
# so unpickling can resolve __main__.TimeFeatures
if "__main__" not in sys.modules:
    sys.modules["__main__"] = types.ModuleType("__main__")
setattr(sys.modules["__main__"], "TimeFeatures", TimeFeatures)
# --- end patch ---

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model_stacking_pipeline.pkl")

model = load_model()

st.title("Solar Power Prediction Dashboard 🌞")

menu = ["Single Prediction", "Batch Prediction"]
choice = st.sidebar.selectbox("Select Option", menu)

if choice == "Single Prediction":
    st.subheader("Single Prediction")
    hour = st.number_input("Hour", min_value=0, max_value=23)
    day = st.number_input("Day", min_value=1, max_value=31)
    month = st.number_input("Month", min_value=1, max_value=12)

    if st.button("Predict"):
        input_data = pd.DataFrame([[hour, day, month]], columns=["Hour", "Day", "Month"])
        prediction = model.predict(input_data)
        st.success(f"Predicted Output: {prediction[0]}")

elif choice == "Batch Prediction":
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        predictions = model.predict(data)
        data["Prediction"] = predictions
        st.write(data)
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
