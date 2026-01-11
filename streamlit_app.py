


import streamlit as st
import os, sys, types, base64, mimetypes
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ========= PATHS =========
#MODEL_PATH = "model_stacking_pipeline.pkl"
IRR_PATH   = "Energy_solar.csv"

st.set_page_config(
    page_title="Tableau de bord de l’énergie solaire",
    page_icon="photo/solar_logo.png",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 6], vertical_alignment="center")

with col1:
    st.image("photo/solar_logo.png", width=160)

with col2:
    st.markdown(
        "<h1 style='margin:0'>Tableau de bord de l’énergie solaire</h1>",
        unsafe_allow_html=True
    )




def inject_css():
    st.markdown("""
    <style>
    /* ================= ROOT THEME ================= */
    :root {
        --accent: #F59E0B;           /* solar amber */
        --accent-soft: #FDE68A;
        --accent-dark: #B45309;
        --bg-glass: rgba(255,255,255,0.65);
        --border-soft: rgba(0,0,0,0.06);
        --shadow-soft: 0 10px 30px rgba(0,0,0,0.08);
        --shadow-hover: 0 16px 40px rgba(0,0,0,0.12);
    }

    /* ================= PAGE ================= */
    .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4 {
        letter-spacing: .2px;
        font-weight: 700;
    }

    h1 {
        background: linear-gradient(90deg, #F59E0B, #FBBF24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ================= SECTION TITLE ================= */
    .section-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin: 1.4rem 0 .6rem;
        padding-top: .6rem;
        border-top: 1px solid var(--border-soft);
        color: #374151;
    }

    /* ================= CARDS ================= */
    .card {
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: var(--shadow-soft);
        transition: all .25s ease;
    }

    .card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-hover);
    }

    /* ================= BUTTONS ================= */
    .stButton > button {
        border-radius: 12px;
        padding: .6rem 1.2rem;
        font-weight: 600;
        background: linear-gradient(135deg, #F59E0B, #FBBF24);
        color: black;
        border: none;
        box-shadow: 0 6px 16px rgba(245,158,11,.35);
        transition: all .2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(245,158,11,.45);
        background: linear-gradient(135deg, #FBBF24, #F59E0B);
    }

    /* ================= METRICS ================= */
    div[data-testid="stMetric"] {
        background: var(--bg-glass);
        border-radius: 16px;
        padding: 14px;
        box-shadow: var(--shadow-soft);
        border: 1px solid var(--border-soft);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #92400E;
    }

    /* ================= TABS ================= */
    [data-baseweb="tab-list"] {
        gap: 6px;
        padding-bottom: 6px;
    }

    [data-baseweb="tab"] {
        border-radius: 12px;
        padding: .45rem .9rem;
        background: rgba(0,0,0,0.03);
        transition: all .2s ease;
        font-weight: 600;
    }

    [data-baseweb="tab"]:hover {
        background: var(--accent-soft);
    }

    [aria-selected="true"] {
        background: linear-gradient(135deg, #F59E0B, #FBBF24) !important;
        color: black !important;
        box-shadow: 0 6px 16px rgba(245,158,11,.35);
    }

    /* ================= DATAFRAMES ================= */
    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--border-soft);
        box-shadow: var(--shadow-soft);
    }

    /* ================= DIVIDERS ================= */
    hr {
        border: none;
        border-top: 1px solid var(--border-soft);
        margin: 1.2rem 0;
    }

    /* ================= SIDEBAR ================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFBEB, #FEF3C7);
        border-right: 1px solid var(--border-soft);
    }

    section[data-testid="stSidebar"] code {
        border-radius: 12px;
    }

    /* ================= SCROLLBAR ================= */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #FBBF24;
        border-radius: 6px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #F59E0B;
    }
    </style>
    """, unsafe_allow_html=True)



# ========= THEME POLISH =========


inject_css()

# ========= RUNTIME VERSIONS (sidebar) =========
try:
    import sklearn, numpy, pandas, joblib as _joblib
    with st.sidebar:
        st.caption("Runtime versions")
        st.code(
            f"Python: 3.11 expected\n"
            f"sklearn={sklearn.__version__}\n"
            f"numpy={numpy.__version__}\n"
            f"pandas={pandas.__version__}\n"
            f"joblib={_joblib.__version__}",
            language="text"
        )
except Exception:
    pass

# ========= FEATURES / TRANSFORMER =========
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

# Make TimeFeatures importable for joblib
if "__main__" not in sys.modules:
    sys.modules["__main__"] = types.ModuleType("__main__")
setattr(sys.modules["__main__"], "TimeFeatures", TimeFeatures)

# ========= MODEL =========
# def _maybe_download_model():
#     url = os.environ.get("MODEL_URL", "").strip()
#     if url and not os.path.exists(MODEL_PATH):
#         import urllib.request
#         st.info("Downloading model…")
#         urllib.request.urlretrieve(url, MODEL_PATH)
#         st.success("Model downloaded.")

# def _maybe_download_model():
#     url = os.environ.get("MODEL_URL", "").strip()
#     if url and not os.path.exists(MODEL_PATH):
#         import requests, time
#         st.info(f"Downloading model from {url} …")
#         for attempt in range(3):
#             try:
#                 with requests.get(url, stream=True, timeout=60) as r:
#                     r.raise_for_status()
#                     with open(MODEL_PATH, "wb") as f:
#                         for chunk in r.iter_content(1024 * 1024):
#                             if chunk:
#                                 f.write(chunk)
#                 st.success("Model downloaded.")
#                 break
#             except Exception as e:
#                 if attempt == 2:
#                     st.error(f"Download failed: {e}")
#                     st.stop()
#                 time.sleep(2)

# @st.cache_resource
# def load_model(path: str, mtime: float):
#     return joblib.load(path)

# if not os.path.exists(MODEL_PATH):
#     _maybe_download_model()
# if not os.path.exists(MODEL_PATH):
#     st.error("Model file not found. Add `model_stacking_pipeline.pkl` or set MODEL_URL.")
#     st.stop()

# model = load_model(MODEL_PATH, os.path.getmtime(MODEL_PATH))

# import os, hashlib, requests, pickle
# from pathlib import Path
# import streamlit as st

# MODEL_URL = "https://github.com/MahamoudAbdi18/FTL-capstone-solar-power project/releases/download/v1.0/model_stacking_pipeline.pk"
# MODEL_SHA256 = "<colle-ici-la-valeur-si-tu-l’as-calculée>"

# def download_file(url: str, dest: Path, sha256: str | None = None):
#     dest.parent.mkdir(parents=True, exist_ok=True)
#     if dest.exists():
#         if sha256:
#             got = hashlib.sha256(dest.read_bytes()).hexdigest()
#             if got != sha256:
#                 dest.unlink()
#             else:
#                 return dest
#         else:
#             return dest
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(dest, "wb") as f:
#             for chunk in r.iter_content(1024*1024):
#                 if chunk:
#                     f.write(chunk)
#     if sha256:
#         assert hashlib.sha256(dest.read_bytes()).hexdigest()==sha256, "Hash mismatch"
#     return dest

# @st.cache_resource
# def load_model():
#     path = download_file(MODEL_URL, Path("artifacts/model_stacking_pipeline.pkl"), MODEL_SHA256)
#     with open(path, "rb") as f:
#         return pickle.load(f)

# model = load_model()

import os, time, hashlib, requests, joblib
from pathlib import Path
import streamlit as st

# ------------- CONFIG -------------
# Where to store the downloaded model (create a folder so the repo stays clean)
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ARTIFACTS_DIR / "model_stacking_pipeline.pkl"

# Try Streamlit secrets first, then environment variable
def get_model_url() -> str:
    url = ""
    try:
        url = st.secrets.get("MODEL_URL", "")
    except Exception:
        pass
    if not url:
        url = os.environ.get("MODEL_URL", "")
    return url.strip()

# ------------- DOWNLOAD -------------
def _maybe_download_model():
    url = get_model_url()
    if not url:
        st.error("MODEL_URL not set in Secrets or environment.")
        st.stop()

    if MODEL_PATH.exists():
        return

    st.info(f"Downloading model from {url} …")
    headers = {"User-Agent": "streamlit-app/1.0 (+https://streamlit.io)"}
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=120, headers=headers, allow_redirects=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", "0") or 0)
                downloaded = 0
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                # simple sanity check
                if total and downloaded < total * 0.9:
                    raise RuntimeError(f"Incomplete download ({downloaded}/{total} bytes)")
            st.success("Model downloaded.")
            return
        except Exception as e:
            if attempt == 2:
                st.error(f"Download failed: {e}")
                st.stop()
            time.sleep(2)

# ------------- LOAD -------------
@st.cache_resource
def load_model(path: str, mtime: float):
    return joblib.load(path)

# ------------- BOOTSTRAP -------------
if not MODEL_PATH.exists():
    _maybe_download_model()
if not MODEL_PATH.exists():
    st.error("Model file not found. Add it to the repo or set MODEL_URL.")
    st.stop()

model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


# ========= HERO =========
left, right = st.columns([1, 1], vertical_alignment="center")
with left:
    st.write("Prédictions à partir des données météo + variables temporelles, prise en charge des fichiers CSV en lot, et évaluation saisonnière du photovoltaïque PV basée sur les données d’irradiance.")
with right:
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Fichier du modèle", "Chargé")
        c2.metric("Variables", f"{len(BASE_COLS)} weather")
        c3.metric("Entrées temporelles", "Heure/Jour/Mois")

st.divider()

# ========= TABS =========
tab1, tab2, tab3, tab4 = st.tabs(["🖊️ Manuel", "📂 CSV en lot", "🔆 Évaluation des panneaux", "👥 Equipe"])

# ---------- TAB 1: Manual ----------
# ---------- TAB 1: Manual ----------
with tab1:
    st.markdown("### Prédiction")
    st.caption("Fournissez un horodatage unique ainsi que les variables météo ci-dessous.")
    st.code(", ".join(BASE_COLS), language="text")

    with st.container():
        with st.form("manual_input_form", clear_on_submit=False, border=False):
            st.markdown('<div class="section-title">Temps</div>', unsafe_allow_html=True)

            dcol1, dcol2 = st.columns(2)
            with dcol1:
                date_val = st.date_input(
                    "Date",
                    value=datetime.now().date()
                )
            with dcol2:
                time_val = st.time_input(
                    "Heure",
                    value=datetime.now().time().replace(
                        minute=0, second=0, microsecond=0
                    )
                )

            dt = pd.to_datetime(f"{date_val} {time_val}")

            st.markdown('<div class="section-title">Météo</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                temperature_2m       = st.number_input("Temperature (°C)", value=25.0)
                relative_humidity_2m = st.number_input("Humidité Relative (%)", 0.0, 100.0, 50.0)
                dew_point_2m         = st.number_input("Point de Rosée (°C)", value=15.0)
            with c2:
                wind_speed_10m     = st.number_input("Vitesse du Vent (km/h)", min_value=0.0, value=10.0)
                wind_direction_10m = st.number_input("Direction du Vent (°)", 0.0, 360.0, 180.0)
                cloud_cover        = st.number_input("Couverture Nuageuse (%)", 0.0, 100.0, 20.0)

            submitted = st.form_submit_button("Prédire")

    # ✅ MUST be aligned with `with st.container():`
    if submitted:
        row = {}

        # --- time ---
        row[RAW_TIME] = dt

        # --- weather ---
        row['temperature_2m (°C)']       = float(temperature_2m)
        row['relative_humidity_2m (%)']  = float(relative_humidity_2m)
        row['dew_point_2m (°C)']         = float(dew_point_2m)
        row['wind_speed_10m (km/h)']     = float(wind_speed_10m)
        row['wind_direction_10m (°)']    = float(wind_direction_10m)
        row['cloud_cover (%)']           = float(cloud_cover)

        # --- safety fill ---
        for col in ALL_FEATURES_WITH_TIME:
            row.setdefault(col, 0 if col in TIME_FEATURES else 0.0)

        X = pd.DataFrame([row], columns=ALL_FEATURES_WITH_TIME)

        try:
            y = model.predict(X)
            pred = float(y[0])

            st.success("Prédiction terminée")
            st.metric("Prédiction (W/m²)", f"{pred:,.2f}")

            out = X.copy()
            out["prediction_W_m2"] = pred
            st.download_button(
                "⬇️ Télécharger CSV",
                out.to_csv(index=False).encode("utf-8"),
                "prediction_single.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ---------- TAB 2: Batch ----------
with tab2:
    st.markdown("### Prédictions en lot à partir d’un CSV")
    st.caption("Votre CSV doit contenir soit une colonne time, soit les trois colonnes Hour, Day, Month, ainsi:")
    st.code(", ".join(BASE_COLS), language="text")

    file = st.file_uploader("Choisir un CSV ", type=["csv"], key="batch_csv")
    if file is not None:
        try:
            df = pd.read_csv(file)
            # Si la colonne time existe, extraire Hour, Day, Month
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
                df["Hour"] = df["time"].dt.hour
                df["Day"] = df["time"].dt.day
                df["Month"] = df["time"].dt.month
            else:
                for c in ["Hour", "Day", "Month"]:
                    if c not in df.columns:
                        df[c] = 0
            for col in ALL_FEATURES_WITH_TIME:
                if col not in df.columns:
                    df[col] = 0.0           
            for c in TIME_FEATURES:
                if c not in df.columns:
                    df[c] = 0
            for col in ALL_FEATURES_WITH_TIME:
                if col not in df.columns:
                    df[col] = 0.0

            X = df[ALL_FEATURES_WITH_TIME]
            with st.spinner("Calcul en cours…"):
                preds = model.predict(X)
            out = df.copy()
            out["prediction_W_m2"] = preds

            st.success(f"Predicted {len(out)} rows")
            st.dataframe(out.head(100), use_container_width=True)
            st.download_button("⬇️ Télécharger predictions.csv",
                               out.to_csv(index=False).encode("utf-8"),
                               "predictions.csv",
                               "text/csv")
        except Exception as e:
            st.error(f"Batch prediction error: {e}")



# ---------- TAB 3: Panel Evaluation (irradiance interne OU CSV utilisateur) ----------
with tab3:
    st.markdown("### Performance photovoltaïque saisonnière et annuelle")
    st.caption("Choisissez la source d’irradiance et l’application sélectionnera automatiquement l’année avec la meilleure couverture ainsi que quatre jours représentatifs des saisons.")

    # === Choix de la source d'irradiance ===
    src_choice = st.radio(
        "Source de données",
        ["Irradiance interne", "Téléverser un CSV"],
        horizontal=True,
        index=0,
        key="irr_src_choice",
    )

    # ---------- Chargement / Préparation des données ----------
    def _clean_time_index(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
        if time_col not in df.columns:
            st.error(f"La colonne temporelle `{time_col}` est introuvable."); st.stop()
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
        return df

    def _load_internal_irradiance() -> pd.DataFrame:
        if "irr_df" in st.session_state and isinstance(st.session_state["irr_df"], pd.DataFrame):
            df = st.session_state["irr_df"].copy()
        else:
            if not os.path.exists(IRR_PATH):
                st.error(f"Irradiance introuvable. Déposez un DataFrame dans st.session_state['irr_df'] ou ajoutez le fichier `{IRR_PATH}`.")
                st.stop()
            df = pd.read_csv(IRR_PATH)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns="Unnamed: 0")
        return _clean_time_index(df, time_col="time")

    irr = None
    tilt_col = None
    horiz_col = None

    if src_choice == "Irradiance interne":
        irr = _load_internal_irradiance()

        # Détection automatique des colonnes
        tilt_candidates  = [c for c in irr.columns if "global_tilted_irradiance" in c.lower()]
        horiz_candidates = [c for c in irr.columns if ("horizontal" in c.lower() and "irradiance" in c.lower())]

        if not tilt_candidates:
            st.error("Aucune colonne ne contient 'global_tilted_irradiance'."); st.stop()
        tilt_col  = tilt_candidates[0]
        horiz_col = horiz_candidates[0] if horiz_candidates else None

    else:  # "Téléverser un CSV"
        up = st.file_uploader("Téléversez votre fichier CSV d’irradiance", type=["csv"], key="user_irr_csv")
        if up is None:
            st.info("En attente d’un fichier CSV…")
            st.stop()
        try:
            df_up = pd.read_csv(up)
        except Exception as e:
            st.error(f"Erreur de lecture du CSV : {e}")
            st.stop()

        # Sélection de la colonne temps si le nom diffère
        time_guess = "time" if "time" in df_up.columns else None
        time_col_sel = st.selectbox("Colonne temporelle", options=list(df_up.columns), index=(list(df_up.columns).index(time_guess) if time_guess in df_up.columns else 0))
        irr = _clean_time_index(df_up, time_col=time_col_sel)

        # Proposer le choix de la colonne inclinée et (optionnel) horizontale
        numeric_cols = [c for c in irr.columns if pd.api.types.is_numeric_dtype(irr[c])]
        # Tentatives auto
        auto_tilt  = next((c for c in numeric_cols if "tilt" in c.lower() and "irr" in c.lower()), None) \
                     or next((c for c in numeric_cols if "global_tilted_irradiance" in c.lower()), None)
        auto_horiz = next((c for c in numeric_cols if "horiz" in c.lower() and "irr" in c.lower()), None)

        tilt_col = st.selectbox(
            "Colonne d’irradiance sur plan incliné (obligatoire)",
            options=numeric_cols,
            index=(numeric_cols.index(auto_tilt) if auto_tilt in numeric_cols else 0)
        )
        horiz_col = st.selectbox(
            "Colonne d’irradiance horizontale (optionnel)",
            options=["(aucune)"] + numeric_cols,
            index=( (numeric_cols.index(auto_horiz) + 1) if auto_horiz in numeric_cols else 0)
        )
        if horiz_col == "(aucune)":
            horiz_col = None

    # ---------- Agrégation horaire & meilleure année ----------
    df_hourly = irr.resample("H").mean(numeric_only=True)
    df_hourly["date"] = df_hourly.index.normalize()
    df_hourly["year"] = df_hourly.index.year
    days_per_year = df_hourly.groupby("year")["date"].nunique().sort_values(ascending=False)
    if days_per_year.empty:
        st.error("Pas de données après agrégation horaire."); st.stop()
    best_year = int(days_per_year.index[0])
    year_df = df_hourly[df_hourly.index.year == best_year].copy()

    # ---------- Sélection de jours représentatifs ----------
    daily_energy = year_df[tilt_col].resample("D").sum(min_count=1).dropna()
    daily_energy.index = daily_energy.index.normalize()

    def in_range(start_mmdd, end_mmdd):
        return (daily_energy.index >= pd.Timestamp(f"{best_year}-{start_mmdd}")) & \
               (daily_energy.index <= pd.Timestamp(f"{best_year}-{end_mmdd}"))

    windows = {
        "Spring (≈Mar 20)": in_range("03-10", "03-30"),
        "Summer (≈Jun 21)": in_range("06-11", "07-01"),
        "Autumn (≈Sep 22)": in_range("09-12", "10-02"),
        "Winter (≈Dec 21)": in_range("12-11", "12-31"),
    }
    qmask = {
        "Spring (≈Mar 20)": ((daily_energy.index >= f"{best_year}-01-01") & (daily_energy.index <= f"{best_year}-03-31")),
        "Summer (≈Jun 21)": ((daily_energy.index >= f"{best_year}-04-01") & (daily_energy.index <= f"{best_year}-06-30")),
        "Autumn (≈Sep 22)": ((daily_energy.index >= f"{best_year}-07-01") & (daily_energy.index <= f"{best_year}-09-30")),
        "Winter (≈Dec 21)": ((daily_energy.index >= f"{best_year}-10-01") & (daily_energy.index <= f"{best_year}-12-31")),
    }

    selected = {}
    for label, wmask in windows.items():
        cand = daily_energy[wmask]
        if not cand.empty:
            selected[label] = cand.idxmax().date()
        else:
            qcand = daily_energy[qmask[label]]
            if not qcand.empty:
                selected[label] = qcand.idxmax().date()

    # Dé-duplication
    used = set()
    for label in list(selected.keys()):
        d = selected[label]
        if d in used:
            wmask = windows[label]
            fallback = daily_energy[wmask].drop(pd.Timestamp(d), errors="ignore")
            if fallback.empty:
                fallback = daily_energy[qmask[label]].drop(pd.Timestamp(d), errors="ignore")
            if not fallback.empty:
                selected[label] = fallback.idxmax().date()
        if label in selected:
            used.add(selected[label])

    ordered_labels = ["Spring (≈Mar 20)", "Summer (≈Jun 21)", "Autumn (≈Sep 22)", "Winter (≈Dec 21)"]
    season_fr = {
        "Spring (≈Mar 20)":  "Équinoxe de printemps",
        "Summer (≈Jun 21)":  "Solstice d’été",
        "Autumn (≈Sep 22)":  "Équinoxe d’automne",
        "Winter (≈Dec 21)":  "Solstice d’hiver",
    }

    selected_dates = [pd.Timestamp(selected[l]) for l in ordered_labels if l in selected]
    if len(selected_dates) < 4:
        top = daily_energy.sort_values(ascending=False)
        uniq = []
        for d in top.index:
            if d.date() not in uniq:
                uniq.append(d.date())
            if len(uniq) == 4: break
        selected_dates = [pd.Timestamp(d) for d in sorted(uniq)]

    # ---------- Paramètres panneaux ----------
    with st.expander("⚙️ Panneaux à comparer (nom, Pmax, dimensions)"):
        nb_panels = st.number_input("Nombre de panneaux", 1, 8, 3, 1, key="nb_panels_tab3")
        defaults = [
            ("TRINA", 620.0, 2.382, 1.134),
            ("LONGi", 550.0, 2.278, 1.134),
            ("JINKO", 460.0, 2.182, 1.029),
        ]
        panels = {}
        for i in range(int(nb_panels)):
            name_d, pmax_d, L_d, W_d = defaults[i] if i < len(defaults) else (f"PANEL{i+1}", 500.0, 1.800, 1.100)
            c1, c2, c3, c4 = st.columns([1.2, 0.9, 0.9, 0.9])
            with c1: name = st.text_input(f"Nom du panneau #{i+1}", value=name_d, key=f"name_{i}_tab3")
            with c2: pmax = st.number_input(f"Pmax #{i+1} (W)", 1.0, value=pmax_d, step=10.0, key=f"pmax_{i}_tab3")
            with c3: L    = st.number_input(f"Longueur #{i+1} (m)", 0.3, value=L_d, step=0.001, format="%.3f", key=f"L_{i}_tab3")
            with c4: W    = st.number_input(f"Largeur #{i+1} (m)",  0.3, value=W_d, step=0.001, format="%.3f", key=f"W_{i}_tab3")
            panels[name.strip() or f"PANEL{i+1}"] = {"Pmax": float(pmax), "dims": (float(L), float(W))}

        # Déduplication des noms
        dedup, seen = {}, set()
        for i, (nm, spec) in enumerate(panels.items(), start=1):
            base = nm or f"PANEL{i}"
            n = base; k = 1
            while n in seen:
                n = f"{base}_{k}"; k += 1
            dedup[n] = spec; seen.add(n)
        panels = dedup

    # ---------- Construction des blocs saisonniers ----------
    blocks, labels_for_blocks = [], []
    for lab, d in zip(ordered_labels, selected_dates):
        day_block = year_df[year_df.index.normalize() == d]
        if not day_block.empty:
            blocks.append(day_block); labels_for_blocks.append(season_fr[lab])

    if len(blocks) < 2:
        st.error("Pas assez de jours distincts pour tracer la figure saisonnière."); st.stop()

    df_sel = pd.concat(blocks, axis=0)
    df_sel["time_series_h"] = np.arange(1, len(df_sel) + 1)

    boundary_idx, cum = [], 0
    for blk in blocks[:-1]:
        cum += len(blk)
        boundary_idx.append(cum)

    # ---------- Densité de puissance par panneau ----------
    for name, p in panels.items():
        area = p["dims"][0] * p["dims"][1]
        eta  = p["Pmax"] / (1000.0 * area)
        df_sel[name] = eta * df_sel[tilt_col]

    # ---------- Graphique saisonnier ----------
    fig = plt.figure(figsize=(11, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    if horiz_col is not None:
        ax1.plot(df_sel["time_series_h"], df_sel[horiz_col], linewidth=2, label="Irradiance horizontale globale")
    ax1.plot(df_sel["time_series_h"], df_sel[tilt_col], linewidth=2, label="Irradiance sur panneaux inclinés")
    for x in boundary_idx: ax1.axvline(x)
    start = 0
    for i, blk in enumerate(blocks):
        end = start + len(blk); center = (start + end) / 2
        ax1.text(center, ax1.get_ylim()[1] * 0.95, labels_for_blocks[i],
                 ha="center", va="top", fontsize=10, fontweight="bold")
        start = end
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.18),
               ncol=2 if horiz_col is not None else 1, frameon=True)
    ax1.set_ylabel("Rayonnement solaire (W/m²)")
    ax1.set_xticks([])

    ax2 = fig.add_subplot(2, 1, 2)
    for idx, (name, _) in enumerate(panels.items()):
        linestyle = "--" if idx % 2 else "-"
        ax2.plot(df_sel["time_series_h"], df_sel[name], linewidth=2, linestyle=linestyle, label=name)
    for x in boundary_idx: ax2.axvline(x)
    ax2.set_ylabel("Puissance de sortie (W/m²)")
    ax2.set_xlabel("Série temporelle (h)")
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.info(f"Année utilisée : **{best_year}** · Jours sélectionnés : " + ", ".join([d.strftime('%Y-%m-%d') for d in selected_dates]))

    # ---------- Énergie annuelle ----------
    annual_density_kWh_m2, annual_module_kWh = {}, {}
    for name, p in panels.items():
        area = p["dims"][0] * p["dims"][1]
        eta  = p["Pmax"] / (1000.0 * area)
        power_density = eta * year_df[tilt_col]
        energy_density_kWh_m2 = power_density.sum() / 1000.0
        annual_density_kWh_m2[name] = energy_density_kWh_m2
        annual_module_kWh[name] = energy_density_kWh_m2 * area

    labels = list(panels.keys())
    vals_module  = [annual_module_kWh[k] for k in labels]
    vals_density = [annual_density_kWh_m2[k] for k in labels]

    fig4 = plt.figure(figsize=(10, 8))
    ax4a = fig4.add_subplot(2, 1, 1)
    bars1 = ax4a.bar(labels, vals_module)
    ax4a.set_ylabel("Énergie annuelle (kWh/an)")
    ax4a.set_title("(a) Module unique")
    for b, v in zip(bars1, vals_module):
        ax4a.text(b.get_x() + b.get_width()/2, b.get_height()*1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax4b = fig4.add_subplot(2, 1, 2)
    bars2 = ax4b.bar(labels, vals_density)
    ax4b.set_ylabel("Énergie annuelle (kWh/m²·an)")
    ax4b.set_title("(b) Par unité de surface")
    ax4b.set_xlabel("Panneau")
    for b, v in zip(bars2, vals_density):
        ax4b.text(b.get_x() + b.get_width()/2, b.get_height()*1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig4, clear_figure=True)

    st.markdown("#### Résumé annuel")
    st.dataframe(
        pd.DataFrame({
            "Panneau": labels,
            "Énergie du module (kWh/an)": [round(x, 1) for x in vals_module],
            "Énergie surfacique (kWh/m²·an)": [round(x, 1) for x in vals_density],
        }),
        use_container_width=True
    )

 # ---------- TAB 4: Team ----------

with tab4:
    st.markdown("### Rencontrez l’équipe")
    st.caption("Cliquez sur une carte pour ouvrir le profil LinkedIn.")

    TEAM = [
        {"name": "Mahmoud Abdi",      "linkedin": "https://www.linkedin.com/in/mahamoud-abdi-abdillahi/", "avatar": "photo/moud.jpg"},
        {"name": "Moustapha Ali",     "linkedin": "https://www.linkedin.com/in/moustaphalifarah/",        "avatar": "photo/mous.jpg"},
        {"name": "Aboubaker Mohamed", "linkedin": "https://www.linkedin.com/in/aboubaker-mohamed-abdi-010114273/", "avatar": "photo/abou.jpg"},
        {"name": "Mohamed Abdirazak Achour", "linkedin": "https://www.linkedin.com/in/mohamed-abdourazak-achour-84b591230?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app/", "avatar": "photo/achour.png"},

    ]

    def _as_data_uri(path: str) -> str | None:
        if not path or not os.path.exists(path): return None
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as f: b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _resolve_img(src: str | None) -> str:
        if not src: return "https://static.streamlit.io/examples/dice.jpg"
        s = src.strip()
        if s.startswith(("data:image/", "http://", "https://")): return s
        return _as_data_uri(s) or "https://static.streamlit.io/examples/dice.jpg"

    _linkedin_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512">
      <rect width="448" height="448" rx="48" ry="48" fill="#0A66C2"/>
      <path fill="#fff" d="M100.3 448H7V148.9h93.3V448zM53.7 108.1C24 108.1 0 84 0 54.3S24 0.6 53.7 0.6s53.7 24.1 53.7 53.7-24.1 53.8-53.7 53.8zM447.9 448h-93.1V302.4c0-34.7-0.7-79.3-48.3-79.3-48.3 0-55.7 37.7-55.7 76.6V448h-93.1V148.9H248v40.8h1.3c13.9-26.4 47.9-54.3 98.6-54.3 105.4 0 124.9 69.4 124.9 159.6V448z"/>
    </svg>
    """.strip()
    _linkedin_data_uri = "data:image/svg+xml;base64," + base64.b64encode(_linkedin_svg.encode("utf-8")).decode("ascii")

    
    def member_card(name: str, avatar: str | None, linkedin: str | None):
        img = _resolve_img(avatar)
        ln  = (linkedin or "").strip()
        clickable_open = f' onclick="window.open(\'{ln}\', \'_blank\')" ' if ln else ""
        html = f"""
        <div class="card" style="text-align:center; cursor:{'pointer' if ln else 'default'};" {clickable_open}>
          <img src="{img}" alt="{name}" style="
           width:100%; max-width:240px; aspect-ratio:4/5;
           object-fit:cover; object-position:center top;
           border-radius:14px; box-shadow:0 4px 16px rgba(0,0,0,.08);" />
          <div style="margin-top:10px;font-weight:700;color:#1f2937">{name}</div>
          {f'<img src="{_linkedin_data_uri}" style="width:30px;height:30px;margin-top:8px;" />' if ln else ""}
        </div>
    """
        st.markdown(html, unsafe_allow_html=True)





    if not TEAM:
        st.info("No team members yet. Fill TEAM list above.")
    else:
        per_row = 4
        for i in range(0, len(TEAM), per_row):
            row = TEAM[i:i+per_row]
            cols = st.columns(len(row), gap="large")
            for col, m in zip(cols, row):
                with col:
                    member_card(m.get("name","(No name)"), m.get("avatar"), m.get("linkedin"))

# ========= FOOTER =========
st.markdown("---")
st.caption("© " + str(datetime.now().year) + " • Solar Power Dashboard • Built with Streamlit")

#*****************************************************************************************************************************************

