
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Paths ===
# Change this path to your CSV if needed
DATA_PATH = Path("/kaggle/input/energy-dataset/Energy_solar.csv")
MODEL_PATH = Path("model_stacking_pipeline.pkl")

TARGET = 'global_tilted_irradiance (W/m²)'
RAW_TIME = 'time'

ORIG_COLUMNS = [
    'temperature_2m (°C)',
    'relative_humidity_2m (%)',
    'dew_point_2m (°C)',
    'apparent_temperature (°C)',
    'wind_speed_10m (km/h)',
    'wind_direction_10m (°)',
    'cloud_cover (%)'
]

TIME_FEATURES = ['Hour', 'Day', 'Month']

class TimeFeatures:
    """Adds Hour/Day/Month from 'time' if present, then drops 'time'."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if RAW_TIME in X.columns:
            X[RAW_TIME] = pd.to_datetime(X[RAW_TIME])
            X['Hour'] = X[RAW_TIME].dt.hour
            X['Day'] = X[RAW_TIME].dt.day
            X['Month'] = X[RAW_TIME].dt.month
            X = X.drop(columns=[RAW_TIME])
        for c in TIME_FEATURES:
            if c not in X.columns:
                X[c] = 0
        return X
    # sklearn API
    def get_params(self, deep=True): return {}
    def set_params(self, **params): return self

def load_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df[RAW_TIME] = pd.to_datetime(df[RAW_TIME])
    df['Hour'] = df[RAW_TIME].dt.hour
    df['Day'] = df[RAW_TIME].dt.day
    df['Month'] = df[RAW_TIME].dt.month

    # remove outliers on train set stats (using full data here for simplicity)
    num_cols = ORIG_COLUMNS + TIME_FEATURES
    q = df[num_cols].quantile([0.01, 0.99])
    lo, hi = q.loc[0.01], q.loc[0.99]
    mask = ~((df[num_cols] < lo) | (df[num_cols] > hi)).any(axis=1)
    return df.loc[mask].reset_index(drop=True)

def build_pipeline():
    rf = RandomForestRegressor(n_estimators=1000, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1)
    svr = SVR(kernel='rbf', gamma='auto', C=10)
    knn = KNeighborsRegressor(n_neighbors=5, p=2, weights='distance')
    gbr = GradientBoostingRegressor(n_estimators=500, max_depth=7, min_samples_leaf=3, random_state=42)
    meta = LinearRegression()

    stack = StackingRegressor(
        estimators=[('RF', rf), ('SVR', svr), ('KNN', knn), ('GBR', gbr)],
        final_estimator=meta,
        n_jobs=-1,
        passthrough=False
    )

    pre = Pipeline([
        ('time_features', TimeFeatures()),
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
    ])

    ct = ColumnTransformer([
        ('prep', pre, ORIG_COLUMNS + TIME_FEATURES + [RAW_TIME])
    ], remainder='drop')

    full = Pipeline([('prep', ct), ('model', stack)])
    return full

def main():
    df = pd.read_csv(DATA_PATH)
    df = load_and_clean(df)

    X = df[[RAW_TIME] + ORIG_COLUMNS + TIME_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print({"MAE": mae, "RMSE": rmse, "R2": r2})

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved pipeline -> {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    main()
